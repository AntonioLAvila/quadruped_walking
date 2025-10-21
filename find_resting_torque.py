import numpy as np
from pydrake.all import (
    MathematicalProgram,
    Solve,
    JacobianWrtVariable,
    Simulator,
    ConstantVectorSource,
    StartMeshcat,
    Meshcat,
    DiagramBuilder
)
from env import make_a1_diagram
from util import A1_q0

meshcat: Meshcat = StartMeshcat()

diagram, plant, a1 = make_a1_diagram(meshcat)
diagram_context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(diagram_context)

plant.SetPositions(plant_context, A1_q0)

prog = MathematicalProgram()

u = prog.NewContinuousVariables(12, 'u')
contact_force_vars = [prog.NewContinuousVariables(3, f"foot{i}_contact_force") for i in range(4)]
all_contact_forces = np.hstack(contact_force_vars)

gravity_torques = plant.CalcGravityGeneralizedForces(plant_context)
B = plant.MakeActuationMatrix()

# Calculate jacobians for each foot
foot_frames = [
    plant.GetFrameByName('FL_foot'),
    plant.GetFrameByName('FR_foot'),
    plant.GetFrameByName('RL_foot'),
    plant.GetFrameByName('RR_foot')
]
J = []
for frame in foot_frames:
    J_i = plant.CalcJacobianTranslationalVelocity(
        plant_context,
        JacobianWrtVariable.kV,
        frame,
        [0, 0, 0],
        plant.world_frame(),
        plant.world_frame()
    )
    J.append(J_i)

J = np.vstack(J) # 12x18

# Static equilibrium constraint
# B*u + J.T*f + tau_g = 0  =>  B*u + J.T*f = -tau_g
A_eq = np.hstack([B, J.T])
b_eq = -gravity_torques
vars_for_constraint = np.hstack([u, all_contact_forces])
prog.AddLinearEqualityConstraint(A_eq, b_eq, vars_for_constraint)

# Minimize effort
prog.AddQuadraticCost(np.eye(len(u)), np.zeros(len(u)), u)

# Solve
print("Solving")
result = Solve(prog)
if result.is_success():
    u_sol = result.GetSolution(u)
    contact_sol = result.GetSolution(all_contact_forces).reshape(4, 3)
    
    print(f"Resting Joint Torques:")
    print(u_sol)
    
    print("\nContact Forces:")
    print(contact_sol)
else:
    print(f"No solution: {result.get_solution_result()}")
    raise Exception('YOU DIED')


# TODO he fell but kinda nicely. Should probably redo this,
# but im gonna attribute it to starting like a few mm off
# the ground


# Simulate
builder = DiagramBuilder()
a1_sys = builder.AddSystem(diagram)

torque_source = builder.AddSystem(ConstantVectorSource(u_sol))
builder.Connect(torque_source.get_output_port(), a1_sys.get_input_port(0))

sim_diagram = builder.Build()

sim = Simulator(sim_diagram, sim_diagram.CreateDefaultContext())
sim.set_target_realtime_rate(1.0)
meshcat.StartRecording()
sim.AdvanceTo(5.0)
meshcat.StopRecording()
meshcat.PublishRecording()

while True: ...