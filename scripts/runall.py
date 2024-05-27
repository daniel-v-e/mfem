import os
import glob
import re
import subprocess  # Needed to capture the command output
import logging

# change dir to examples
os.chdir("examples")

# Configure the logging module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command, logfile):
    logging.info(f"Running command: {command}")
    
    # Capture the output and error of the command
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Log the output to the specified logfile
    with open(logfile, 'a') as f:
        f.write(result.stdout.decode())
        f.write(result.stderr.decode())

def extract_sample_run(line):
    match = re.search(r"(ex\d+ -m \.\./data/[\w-]+\.mesh)", line)
    if match:
        return match.group(1)
    match = re.search(r"(ex\d+ -o \d+)", line)
    if match:
        return match.group(1)
    return None

def custom_sort(filename):
    match = re.search(r'ex(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')

def main():
    filepaths = sorted(glob.glob('*.cpp'), key=custom_sort)
    filepaths = [f for f in filepaths if not f.endswith('p.cpp')]
    filepaths = ['ex3.cpp', 'ex34.cpp']
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        basename, _ = os.path.splitext(filename)
        
        # Run make command
        run_command(f"make {basename}", f"{basename}.make.log")
        
        # Extract and run the sample command from the cpp file
        command = None
        with open(filepath, 'r') as file:
            for line in file:
                command = extract_sample_run(line)
                if command:
                    command = f"./{command}"
                    run_command(command, f"{basename}.log")
                    break
        
        if not command:
            logging.warning(f"Failed to find the sample command in {filename}")
            continue

        if os.path.exists("sol.gf"):
            os.rename("sol.gf", f"{basename}.gf")
        if os.path.exists("mesh.mesh"):
            os.rename("mesh.mesh", f"{basename}.mesh")
        if os.path.exists("refined.mesh"):
            os.rename("refined.mesh", f"{basename}_refined.mesh")
        
        mesh_name = command.split("-m")[1].split()[0] if "-m" in command else "mesh.mesh"
        
        # Run the glvis command
        run_command(f"~/glvis-4.2/glvis -m {mesh_name} -g {basename}.gf", f"{basename}.glvis.log")

        # delete mesh.mesh, refined.mesh, sol.gf
        if os.path.exists("sol.gf"):
            os.remove("sol.gf")
        if os.path.exists("mesh.mesh"):
            os.remove("mesh.mesh")
        if os.path.exists("refined.mesh"):
            os.remove("refined.mesh")

if __name__ == "__main__":
    main()
