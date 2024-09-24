import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initial values
num = 400  # number of cells in a cable
v = np.zeros(num)
tmp = np.zeros(num)  # temporary array to solve the PDE
h = np.ones(num)
f = 0.9 * np.ones(num)
stim = np.zeros(num)

# Constants
tauso = 15
taufi = 0.8
tauh1 = 4.8
tauh2 = 10.0
tausi = 4
tauf1 = 100
tauf2 = 30

# Constants for the PDE
dfu = 0.0005  # diffusion coefficient
dx = 0.015  # cell size
dt = 0.1  # time step (0.1 ms)

pcl = 200  # pacing cycle length (ms)
itr = 10  # the number of beats
tmax = pcl * itr  # total time

# Convert to integer
tnmax = round(tmax / dt)
pcln = round(pcl / dt)
durn = round(1.0 / dt)

# Array for results
resultv = []

# Main loop
for tn in range(tnmax + 1):

    # Stimulation
    if tn % pcln < durn:
        stim[:5] = 0.3
    elif tn % pcln == durn:
        stim[:5] = 0

    minf = (v / 0.2)**6 / (1 + (v / 0.2)**6)
    hinf = 1 / (1 + (v / 0.1)**6)
    dinf = (v / 0.4)**4 / (1 + (v / 0.4)**4)
    finf = 1 / (1 + (v / 0.1)**4)

    tauh = tauh1 + tauh2 * np.exp(-20 * (v - 0.1)**2)
    tauf = tauf2 + (tauf1 - tauf2) * v**3

    jfi = h * minf * (v - 1.3) / taufi
    jsi = f * dinf * (v - 1.4) / tausi
    jso = (1 - np.exp(-4 * v)) / tauso
    ion = -(jfi + jsi + jso - stim)

    # Update variables
    v += ion * dt
    h += (hinf - h) * dt / tauh
    f += (finf - f) * dt / tauf

    # Non-flux boundary condition
    v[0] = v[2]
    v[num - 1] = v[num - 3]

    # Solve diffusion equation
    for c in range(1, num - 1):
        tmp[c] = v[c] + (v[c - 1] + v[c + 1] - 2 * v[c]) * dfu * dt / (dx * dx)

    v = np.copy(tmp)

    if tn % 10 == 0:
        resultv.append(v.copy())

# Space-time plot
resultv = np.array(resultv)

plt.figure(figsize=(10, 6))
plt.imshow(resultv.T, aspect='auto', cmap='jet', extent=[0, resultv.shape[0], 0, num], origin='lower')
plt.colorbar(label='Voltage (v)')
plt.title('Space-Time Plot of Voltage (v)')
plt.xlabel('Time (steps)')
plt.ylabel('Space (cells)')
plt.clim(0, 1.2)
plt.show()

import numpy as np
import matplotlib.pyplot as plt


#Q3
# Function to run the simulation with a modified jsi
def run_simulation_jsi(pcl, jsi_factor, num_cells=400, total_beats=10, dt=0.1, stim_strength=0.3):
    # Initial values
    v = np.zeros(num_cells)
    h = np.ones(num_cells)
    f = 0.9 * np.ones(num_cells)
    stim = np.zeros(num_cells)

    # Constants
    tauso = 15
    taufi = 0.8
    tauh1 = 4.8
    tauh2 = 10.0
    tausi = 4
    tauf1 = 100
    tauf2 = 30

    # PDE constants
    dfu = 0.0005  # diffusion coefficient
    dx = 0.015  # cell size

    # Simulation time parameters
    tmax = pcl * total_beats
    tnmax = round(tmax / dt)
    pcln = round(pcl / dt)
    durn = round(1.0 / dt)  # duration of the stimulus

    # Store results for the voltage (space-time plot)
    resultv = []

    # Main loop over time steps
    for tn in range(tnmax + 1):
        # Stimulation
        if tn % pcln < durn:
            stim[:5] = stim_strength
        else:
            stim[:5] = 0

        # Update equations
        minf = (v / 0.2)**6 / (1 + (v / 0.2)**6)
        hinf = 1 / (1 + (v / 0.1)**6)
        dinf = (v / 0.4)**4 / (1 + (v / 0.4)**4)
        finf = 1 / (1 + (v / 0.1)**4)

        tauh = tauh1 + tauh2 * np.exp(-20 * (v - 0.1)**2)
        tauf = tauf2 + (tauf1 - tauf2) * v**3

        # Modify jsi with the factor
        jfi = h * minf * (v - 1.3) / taufi
        jsi = jsi_factor * f * dinf * (v - 1.4) / tausi  # Modify jsi here with the factor
        jso = (1 - np.exp(-4 * v)) / tauso
        ion = -(jfi + jsi + jso - stim)

        # Update variables
        v += ion * dt
        h += (hinf - h) * dt / tauh
        f += (finf - f) * dt / tauf

        # Non-flux boundary conditions
        v[0] = v[2]
        v[-1] = v[-3]

        # Solve diffusion equation
        for c in range(1, num_cells - 1):
            v[c] += (v[c - 1] + v[c + 1] - 2 * v[c]) * dfu * dt / (dx * dx)

        # Record voltage for surface plot
        if tn % 10 == 0:
            resultv.append(v.copy())

    return np.array(resultv)

# Run simulation with PCL = 200 ms and jsi multiplied by 0.8 (reduced Ca current)
resultv_jsi_0_8 = run_simulation_jsi(pcl=200, jsi_factor=0.8)

# Run simulation with PCL = 200 ms and jsi multiplied by 1.2 (increased Ca current)
resultv_jsi_1_2 = run_simulation_jsi(pcl=200, jsi_factor=1.2)

# Plot the results for jsi = 0.8
plt.figure(figsize=(10, 6))
plt.imshow(resultv_jsi_0_8.T, aspect='auto', cmap='jet', extent=[0, resultv_jsi_0_8.shape[0], 0, 400], origin='lower')
plt.colorbar(label='Voltage (v)')
plt.title('Space-Time Plot with jsi = 0.8 (Reduced Ca Current)')
plt.xlabel('Time (steps)')
plt.ylabel('Space (cells)')
plt.clim(0, 1.2)
plt.show()

# Plot the results for jsi = 1.2
plt.figure(figsize=(10, 6))
plt.imshow(resultv_jsi_1_2.T, aspect='auto', cmap='jet', extent=[0, resultv_jsi_1_2.shape[0], 0, 400], origin='lower')
plt.colorbar(label='Voltage (v)')
plt.title('Space-Time Plot with jsi = 1.2 (Increased Ca Current)')
plt.xlabel('Time (steps)')
plt.ylabel('Space (cells)')
plt.clim(0, 1.2)
plt.show()


#Q4
def run_simulation_jfi(pcl, jfi_factor, num_cells=400, total_beats=10, dt=0.1, stim_strength=0.3):
    # Initial values
    v = np.zeros(num_cells)
    h = np.ones(num_cells)
    f = 0.9 * np.ones(num_cells)
    stim = np.zeros(num_cells)

    # Constants
    tauso = 15
    taufi = 0.8
    tauh1 = 4.8
    tauh2 = 10.0
    tausi = 4
    tauf1 = 100
    tauf2 = 30

    # PDE constants
    dfu = 0.0005  # diffusion coefficient
    dx = 0.015  # cell size

    # Simulation time parameters
    tmax = pcl * total_beats
    tnmax = round(tmax / dt)
    pcln = round(pcl / dt)
    durn = round(1.0 / dt)  # duration of the stimulus

    # Store results for the voltage (space-time plot)
    resultv = []

    # Main loop over time steps
    for tn in range(tnmax + 1):
        # Stimulation
        if tn % pcln < durn:
            stim[:5] = stim_strength
        else:
            stim[:5] = 0

        # Update equations
        minf = (v / 0.2)**6 / (1 + (v / 0.2)**6)
        hinf = 1 / (1 + (v / 0.1)**6)
        dinf = (v / 0.4)**4 / (1 + (v / 0.4)**4)
        finf = 1 / (1 + (v / 0.1)**4)

        tauh = tauh1 + tauh2 * np.exp(-20 * (v - 0.1)**2)
        tauf = tauf2 + (tauf1 - tauf2) * v**3

        # Modify jfi with the factor
        jfi = jfi_factor * h * minf * (v - 1.3) / taufi  # Modify jfi here with the factor
        jsi = f * dinf * (v - 1.4) / tausi  # Keep jsi at its original value (factor 1)
        jso = (1 - np.exp(-4 * v)) / tauso
        ion = -(jfi + jsi + jso - stim)

        # Update variables
        v += ion * dt
        h += (hinf - h) * dt / tauh
        f += (finf - f) * dt / tauf

        # Non-flux boundary conditions
        v[0] = v[2]
        v[-1] = v[-3]

        # Solve diffusion equation
        for c in range(1, num_cells - 1):
            v[c] += (v[c - 1] + v[c + 1] - 2 * v[c]) * dfu * dt / (dx * dx)

        # Record voltage for surface plot
        if tn % 10 == 0:
            resultv.append(v.copy())

    return np.array(resultv)

# Run simulation with PCL = 200 ms and jfi multiplied by 0.5 (reduced Na current)
resultv_jfi_0_5 = run_simulation_jfi(pcl=200, jfi_factor=0.5)

# Run simulation with PCL = 200 ms and jfi multiplied by 0.1 (further reduced Na current)
resultv_jfi_0_1 = run_simulation_jfi(pcl=200, jfi_factor=0.1)

# Plot the results for jfi = 0.5
plt.figure(figsize=(10, 6))
plt.imshow(resultv_jfi_0_5.T, aspect='auto', cmap='jet', extent=[0, resultv_jfi_0_5.shape[0], 0, 400], origin='lower')
plt.colorbar(label='Voltage (v)')
plt.title('Space-Time Plot with jfi = 0.5 (Reduced Na Current)')
plt.xlabel('Time (steps)')
plt.ylabel('Space (cells)')
plt.clim(0, 1.2)
plt.show()

# Plot the results for jfi = 0.1
plt.figure(figsize=(10, 6))
plt.imshow(resultv_jfi_0_1.T, aspect='auto', cmap='jet', extent=[0, resultv_jfi_0_1.shape[0], 0, 400], origin='lower')
plt.colorbar(label='Voltage (v)')
plt.title('Space-Time Plot with jfi = 0.1 (Significantly Reduced Na Current)')
plt.xlabel('Time (steps)')
plt.ylabel('Space (cells)')
plt.clim(0, 1.2)
plt.show()

# Function to compute conduction velocity
def calculate_conduction_velocity(resultv, dt, dx):
    # Measure the time taken for the action potential to travel from one cell to another
    num_steps, num_cells = resultv.shape
    cell_1 = 50  # Starting cell
    cell_2 = 150  # Farther cell to measure the wave propagation

    # Find the time index where the action potential crosses a threshold (e.g., v = 0.1) for each cell
    threshold = 0.1
    time_cell_1 = np.argmax(resultv[:, cell_1] > threshold)
    time_cell_2 = np.argmax(resultv[:, cell_2] > threshold)

    # Calculate the time difference and convert it to milliseconds
    time_diff_ms = (time_cell_2 - time_cell_1) * dt

    # Calculate the distance between the cells in cm (dx is in cm)
    distance_cm = (cell_2 - cell_1) * dx

    # Conduction velocity in cm/s
    conduction_velocity = distance_cm / (time_diff_ms / 1000)  # Convert ms to seconds
    return conduction_velocity

# Calculate conduction velocity for jfi = 0.5
cv_jfi_0_5 = calculate_conduction_velocity(resultv_jfi_0_5, dt=0.1, dx=0.015)
print(f'Conduction velocity for jfi = 0.5: {cv_jfi_0_5:.2f} cm/s')

# Calculate conduction velocity for jfi = 0.1
cv_jfi_0_1 = calculate_conduction_velocity(resultv_jfi_0_1, dt=0.1, dx=0.015)
print(f'Conduction velocity for jfi = 0.1: {cv_jfi_0_1:.2f} cm/s')












#Q5
def calculate_apd(voltage_data, dt, threshold=0.1):
    num_cells = voltage_data.shape[1]
    apd_values = []

    for cell in range(num_cells):
        v_cell = voltage_data[:, cell]
        # Find when the action potential crosses the threshold (both depolarization and repolarization)
        above_threshold = v_cell > threshold
        crossing_times = np.where(np.diff(above_threshold.astype(int)))[0]  # Find the crossing indices
        # We expect two crossings per beat (one for depolarization, one for repolarization)
        if len(crossing_times) >= 2:
            apd = (crossing_times[1] - crossing_times[0]) * dt  # Time between depolarization and repolarization
            apd_values.append(apd)
        else:
            apd_values.append(0)  # If no valid crossing is found, set APD to 0

    return np.array(apd_values)

# Extract last two beats from resultv for both PCL = 200 ms and PCL = 140 ms
def extract_last_two_beats(resultv, pcln, dt, beat_interval=2):
    # Assuming each beat has duration pcln
    num_steps = resultv.shape[0]
    start_index = num_steps - pcln * beat_interval  # Start of the second last beat
    return resultv[start_index:], dt

# Simulate for PCL = 200 ms and PCL = 140 ms
pcl_200 = 200  # PCL = 200 ms
pcl_140 = 140  # PCL = 140 ms
dt = 0.1  # Time step in ms

# Example voltage data from the previous simulations (replace with your actual resultv_200 and resultv_140)
# Assuming you have resultv_200 and resultv_140 from your previous simulations

# Extract the last two beats for both PCL cases
last_two_beats_200, dt_200 = extract_last_two_beats(resultv_jsi_1_2, pcln=round(pcl_200 / dt), dt=dt)
last_two_beats_140, dt_140 = extract_last_two_beats(resultv_jsi_1_2, pcln=round(pcl_140 / dt), dt=dt)

# Calculate APD for both last two beats
apd_200_last_beat = calculate_apd(last_two_beats_200, dt_200)
apd_140_last_beat = calculate_apd(last_two_beats_140, dt_140)

# Space (cells 1 to 400)
cells = np.arange(1, 401)

# Plot APD vs Space for both PCL cases
plt.figure(figsize=(12, 6))

# Plot for PCL = 200 ms
plt.plot(cells, apd_200_last_beat, label='PCL = 200 ms (Last Beat)', color='blue')
plt.plot(cells, apd_140_last_beat, label='PCL = 140 ms (Last Beat)', color='red')

plt.title('APD vs Space for PCL = 200 ms and PCL = 140 ms')
plt.xlabel('Space (Cell number)')
plt.ylabel('APD (ms)')
plt.legend()
plt.show()


#Q6
def run_simulation_stim(pcl, stim_strength, num_cells=400, total_beats=10, dt=0.1):
    # Initial values
    v = np.zeros(num_cells)
    h = np.ones(num_cells)
    f = 0.9 * np.ones(num_cells)
    stim = np.zeros(num_cells)
    tauso = 15
    taufi = 0.8
    tauh1 = 4.8
    tauh2 = 10.0
    tausi = 4
    tauf1 = 100
    tauf2 = 30
    dfu = 0.0005
    dx = 0.015
    tmax = pcl * total_beats
    tnmax = round(tmax / dt)
    pcln = round(pcl / dt)
    durn = round(1.0 / dt)
    resultv = []
    for tn in range(tnmax + 1):
        # Stimulation
        if tn % pcln < durn:
            stim[:5] = stim_strength
        else:
            stim[:5] = 0
        minf = (v / 0.2)**6 / (1 + (v / 0.2)**6)
        hinf = 1 / (1 + (v / 0.1)**6)
        dinf = (v / 0.4)**4 / (1 + (v / 0.4)**4)
        finf = 1 / (1 + (v / 0.1)**4)

        tauh = tauh1 + tauh2 * np.exp(-20 * (v - 0.1)**2)
        tauf = tauf2 + (tauf1 - tauf2) * v**3

        jfi = h * minf * (v - 1.3) / taufi
        jsi = f * dinf * (v - 1.4) / tausi
        jso = (1 - np.exp(-4 * v)) / tauso
        ion = -(jfi + jsi + jso - stim)

        # Update variables
        v += ion * dt
        h += (hinf - h) * dt / tauh
        f += (finf - f) * dt / tauf

        # Non-flux boundary conditions
        v[0] = v[2]
        v[-1] = v[-3]

        # Solve diffusion equation
        for c in range(1, num_cells - 1):
            v[c] += (v[c - 1] + v[c + 1] - 2 * v[c]) * dfu * dt / (dx * dx)

        # Record voltage for surface plot
        if tn % 10 == 0:
            resultv.append(v.copy())

    return np.array(resultv)

# List of stimulation currents to test
stim_values = [0.3, 0.25, 0.22, 0.2, 0.18, 0.15, 0.14, 0.09]
# Run the simulation for each stimulation value
for stim_strength in stim_values:
    print(f"Running simulation with stim_strength = {stim_strength}")
    resultv = run_simulation_stim(pcl=200, stim_strength=stim_strength)

    # Plot the results to observe if an action potential wave was generated
    plt.figure(figsize=(10, 6))
    plt.imshow(resultv.T, aspect='auto', cmap='jet', extent=[0, resultv.shape[0], 0, 400], origin='lower')
    plt.colorbar(label='Voltage (v)')
    plt.title(f'Space-Time Plot with Stimulation Current = {stim_strength}')
    plt.xlabel('Time (steps)')
    plt.ylabel('Space (cells)')
    plt.clim(0, 1.2)
    plt.show()
    if np.max(resultv[:, -1]) > 0.1:
        print(f"Action potential wave successfully generated with stim_strength = {stim_strength}")
    else:
        print(f"Action potential wave failed with stim_strength = {stim_strength}")