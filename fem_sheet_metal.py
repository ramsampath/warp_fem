# fem_sheet_metal.py
# Warp-based sheet metal bending with optional plasticity on bending edges.

import math
import os
from enum import Enum

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.sim
import warp.sim.render


class IntegratorType(Enum):
    EULER = "euler"
    XPBD = "xpbd"
    VBD = "vbd"
    def __str__(self): return self.value

@wp.kernel
def damp_vec3(v: wp.array(dtype=wp.vec3), factor: float):
    i = wp.tid()
    v[i] = v[i] * factor


@wp.kernel
def clamp_speed(v: wp.array(dtype=wp.vec3), vmax: float):
    i = wp.tid()
    s = wp.length(v[i])
    if s > vmax:
        v[i] = v[i] * (vmax / (s + 1.0e-8))



# ---- plasticity kernel: still not completely tested 
@wp.func
def dihedral_angle(p_o0: wp.vec3, p_o1: wp.vec3, p_v1: wp.vec3, p_v2: wp.vec3) -> float:
    # edge is v1--v2; triangles (o0, v1, v2) and (o1, v2, v1)
    e  = wp.normalize(p_v2 - p_v1)
    n0 = wp.normalize(wp.cross(p_v1 - p_o0, p_v2 - p_o0))
    n1 = wp.normalize(wp.cross(p_v2 - p_o1, p_v1 - p_o1))
    # signed angle around shared edge
    sin_t = wp.dot(wp.cross(n0, n1), e)
    cos_t = wp.clamp(wp.dot(n0, n1), -1.0, 1.0)
    return wp.atan2(sin_t, cos_t)

@wp.kernel
def set_body_q(q: wp.array(dtype=wp.transform), index: int, value: wp.transform):
    if wp.tid() == 0:
        q[index] = value


@wp.kernel
def set_body_qd(qd: wp.array(dtype=wp.spatial_vector), index: int, value: wp.spatial_vector):
    if wp.tid() == 0:
        qd[index] = value


@wp.kernel
def plastic_update_bend(
    edge_indices: wp.array(dtype=int, ndim=2),   # <-- ndim=2
    particle_q: wp.array(dtype=wp.vec3),
    rest_angle: wp.array(dtype=float),
    yield_angle: float,
    creep: float
):
    i = wp.tid()

    # edge_indices has shape (N, 4): [o0, o1, v1, v2]
    o0 = edge_indices[i, 0]
    o1 = edge_indices[i, 1]
    v1 = edge_indices[i, 2]
    v2 = edge_indices[i, 3]

    a = dihedral_angle(particle_q[o0], particle_q[o1], particle_q[v1], particle_q[v2])
    r = rest_angle[i]
    if wp.abs(a - r) > yield_angle:
        rest_angle[i] = r + creep * (a - r)

@wp.kernel
def translate_all(q: wp.array(dtype=wp.vec3), delta: wp.vec3):
    i = wp.tid()
    q[i] = q[i] + delta




class fem_sheet_metal_sim:
    """Sheet metal forming simulation with optional plasticity."""
    def __init__(
        self,
        stage_path=None,                        # path to save USD scene (or None to disable)",
        integrator: IntegratorType = IntegratorType.XPBD,
        width=80, height=40,
        cell=0.01,                               # 1cm mesh spacing
        punch_shape="sphere",                    # "sphere" or "box"
        punch_radius=0.08,                       # 8 cm radius (or half-extent for box on X/Z)
        punch_depth=0.08,                        # travel distance down
        punch_speed=0.15,                        # m/s downward
        enable_plasticity=True,
        yield_deg=2.0,                           # ~2 degrees yield
        creep=0.05                               # plastic creep factor in [0,1]
    ):
        self.integrator_type = integrator
        self.sim_width  = width
        self.sim_height = height
        self.cell = cell

        # timing
        fps = 60
        self.substeps = 64
        self.frame_dt = 1.0 / fps
        self.sim_dt = self.frame_dt / self.substeps
        self.t = 0.0
        self.profiler = {}

        

        b = wp.sim.ModelBuilder()

        # ---- parameters for supports ----
        gap = 0.15
        hx, hy, hz = 0.2, 0.03, 0.5
        y0 = 0.0
        y_top = y0 + hy  # top of supports

        # ---- contact thickness/skin scale with mesh density ----
        self._thickness = max(0.4 * self.cell, 0.0015)   # ~0.4×cell, >=1.5 mm
        self._skin      = max(0.3 * self.cell, 0.0010)   # ~0.3×cell, >=1.0 mm

        # sheet’s initial vertical gap above supports: must exceed thickness+skin
        clearance = self._thickness + self._skin + 0.001  # add 1 mm extra

        b.add_cloth_grid(
            pos=wp.vec3(-0.5, y_top + clearance, 0.2),
            rot=wp.quat_from_axis_angle(wp.vec3(1,0,0), -math.pi*0.5),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.sim_width, dim_y=self.sim_height,
            cell_x=self.cell, cell_y=self.cell,
            mass=0.05,
            fix_left=False, fix_right=False, fix_top=False, fix_bottom=False,
            # Reduced stiffness for stability
            tri_ke=2.5e4, tri_ka=2.5e4, tri_kd=6.0e3,
            edge_ke=500, edge_kd=1.5e3
            #edge_ke=7.5e3, edge_kd=2.5e3,
        )


        # left support
        b.add_shape_box(
            body=-1,
            pos=wp.vec3(-gap*0.5 - hx, y0, 0.0),
            rot=wp.quat_identity(),
            hx=hx, hy=hy, hz=hz,
            ke=2.0e5, kd=2.0e4, kf=2.0e3, mu=0.6
        )
        # right support
        b.add_shape_box(
            body=-1,
            pos=wp.vec3(+gap*0.5 + hx, y0, 0.0),
            rot=wp.quat_identity(),
            hx=hx, hy=hy, hz=hz,
            ke=2.0e5, kd=2.0e4, kf=2.0e3, mu=0.6
        )


        # --- punch: kinematic rigid body ---
        self.punch_body = b.add_body(origin=wp.transform((0.0, 0.10, 0.0), wp.quat_identity()), m=0.0)
        if punch_shape == "sphere":
            b.add_shape_sphere(body=self.punch_body, radius=punch_radius,
                            #ke=2.0e5, kd=2.0e4, kf=2.0e3, mu=0.4)
                            ke=3.0e4, kd=2.0e4, kf=5.0e2, mu=0.2)
        else:
            b.add_shape_box(body=self.punch_body, hx=punch_radius, hy=punch_radius, hz=punch_radius,
                            #ke=2.0e5, kd=2.0e4, kf=2.0e3, mu=0.4)
                            ke=3.0e4, kd=2.0e4, kf=5.0e2, mu=0.2)
    

        self.integrator_type == IntegratorType.VBD

        if self.integrator_type == IntegratorType.VBD:
            b.color()

        # finalize
        self.model = b.finalize()

        st = self.model.state()
 
        #self.model.gravity = wp.vec3(0.0, -0., 0.0)

        self.model.ground = False
        #self.model.gravity = wp.vec3(0.0, 0.0, 0.0)
        self.model.gravity = (0.0, -9.81, 0.0)


        self.model.ground  = False

        self._thickness = max(0.4 * self.cell, 0.0015)
        self._skin      = max(0.3 * self.cell, 0.0010)
        self.model.particle_radius.fill_(self._thickness)
        self.model.soft_contact_margin = self._skin

        self.model.soft_contact_ke = 5.0e4     
        self.model.soft_contact_kd = 3.0e4    
        self.model.soft_contact_kf = 5.0e2
        self.model.soft_contact_mu = 0.20     

        self._punch_d_prev = 0.0


        # integrator
        if self.integrator_type == IntegratorType.EULER:
            self.integrator = wp.sim.SemiImplicitIntegrator()
        elif self.integrator_type == IntegratorType.XPBD:
            self.integrator = wp.sim.XPBDIntegrator(iterations=10)
        else:
            self.integrator = wp.sim.VBDIntegrator(self.model, iterations=32)

        #self.integrator = wp.sim.SemiImplicitIntegrator()

        #self.integrator = wp.sim.VBDIntegrator(self.model, iterations=32)  # 24–32 if needed
        self.substeps   = 64 


        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        q = self.state_0.particle_q.numpy()


        # renderer
        self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=40.0) if stage_path else None
        #self.renderer2 = wp.sim.render.SimRenderer(self.model, "Orig.usd", scaling=40.0) if stage_path else None


        # motion params
        self.punch_shape = punch_shape
        self.punch_start = wp.vec3(0.0, 0.1, 0.0)
        self.punch_depth = punch_depth
        self.punch_speed = punch_speed

        # plasticity
        self.enable_plasticity = enable_plasticity
        self.yield_angle = math.radians(yield_deg)
        self.creep = float(creep)

        self.use_cuda_graph = wp.get_device().is_cuda
        print (f"Using CUDA graph: {self.use_cuda_graph}")

        self.use_cuda_graph = False  # TEMP DISABLE FOR DEBUGGING
        
        if self.use_cuda_graph:
            with wp.ScopedCapture() as cap:
                self._simulate_one_frame()
            self.graph = cap.graph



    def _animate_punch(self, t):
        d_goal = min(self.punch_speed * t, self.punch_depth)

        # limit increment this substep to less than ~25% of contact skin+thickness
        max_dl = 0.15 * (self._skin + 2.0*self._thickness)   # was 0.25

        d = self._punch_d_prev + max(-max_dl, min(max_dl, d_goal - self._punch_d_prev))
        self._punch_d_prev = d

        pos = self.punch_start - wp.vec3(0.0, d, 0.0)
        xform = wp.transform(pos, wp.quat_identity())

        # write pose into both states so the swap never loses it
        for q in (self.state_0.body_q, self.state_1.body_q):
            wp.launch(set_body_q, dim=1, inputs=[q, self.punch_body, xform], device=wp.get_device())



    def _plastic_step(self):
        if not self.enable_plasticity or self.model.edge_count == 0:
            return
        wp.launch(
            kernel=plastic_update_bend,
            dim=self.model.edge_count,
            inputs=[
                self.model.edge_indices,
                self.state_0.particle_q,
                self.model.edge_rest_angle,
                float(self.yield_angle),
                float(self.creep),
            ],
            device=wp.get_device()
        )


    def _simulate_one_frame(self):
        for s in range(self.substeps):
            self.state_0.clear_forces()

            # substep time
            tau = self.t + (s + 1) * self.sim_dt

            # move punch first
            self._animate_punch(tau)

            # rebuild contacts AFTER moving the kinematic body
            wp.sim.collide(self.model, self.state_0)

            # integrate one substep
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

            # gentle damping & clamp on particle velocities (use particle_qd)
            dev = wp.get_device()
            vel = getattr(self.state_1, "particle_qd", None)
            if vel is not None:
                # Increased damping and more aggressive velocity clamp
                wp.launch(damp_vec3,   dim=self.model.particle_count, inputs=[vel, 0.90], device=dev)
                wp.launch(clamp_speed, dim=self.model.particle_count, inputs=[vel, 0.5], device=dev)

            # plasticity update after velocity update
            #self._plastic_step()

            # ensure output state has same kinematic pose
            self._animate_punch(tau)

            # swap buffers
            self.state_0, self.state_1 = self.state_1, self.state_0



    def step(self):
        print("Time: %.3f" % self.t)
        if int(self.t / self.frame_dt) % 30 == 0:
            q_np = self.state_0.body_q.numpy() 
            if getattr(q_np.dtype, "fields", None) and "p" in q_np.dtype.fields:
                y = float(q_np["p"][self.punch_body, 1])
            else:
                # plain ndarray, usually (N,7): [px,py,pz,qx,qy,qz,qw]
                y = float(q_np[self.punch_body, 1])
            print(f"t={self.t:.3f}  punch y={y:.4f}")

        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate_one_frame()
        self.t += self.frame_dt


    def render(self):
        if not self.renderer:
            return
        self.renderer.begin_frame(self.t)
        self.renderer.render(self.state_0)
        self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stage_path", type=lambda x: None if x == "None" else str(x), default="sheet_metal.usda")
    parser.add_argument("--num_steps", type=int, default=240)
    parser.add_argument("--integrator", type=IntegratorType, choices=list(IntegratorType), default=IntegratorType.XPBD)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--height", type=int, default=40)
    parser.add_argument("--cell", type=float, default=0.01)
    parser.add_argument("--punch_shape", choices=["sphere", "box"], default="sphere")
    parser.add_argument("--punch_radius", type=float, default=0.02)
    parser.add_argument("--punch_depth", type=float, default=0.08)
    parser.add_argument("--punch_speed", type=float, default=0.15)
    parser.add_argument("--plasticity", action="store_true")
    parser.add_argument("--yield_deg", type=float, default=2.0)
    parser.add_argument("--creep", type=float, default=0.05)
    args = parser.parse_args()

    with wp.ScopedDevice(args.device):
        ex = fem_sheet_metal_sim(
            stage_path=args.stage_path,
            integrator=args.integrator,
            width=args.width, height=args.height, cell=args.cell,
            punch_shape=args.punch_shape, punch_radius=args.punch_radius,
            punch_depth=args.punch_depth, punch_speed=args.punch_speed,
            enable_plasticity=args.plasticity, yield_deg=args.yield_deg, creep=args.creep
        )
        for _ in range(args.num_steps):
            ex.step()
            ex.render()

        if ex.renderer:
            ex.renderer.save()
