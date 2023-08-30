import os
import glob
import re  # Import the regex module

# change dir to examples
os.chdir("examples")

def run_command(command):
    print(f"\nRunning command: {command}")
    os.system(command)

def extract_sample_run(line):
    # Use regex to match the pattern: ex* -m ../data/*.mesh
    match = re.search(r"(ex\d+ -m \.\./data/[\w-]+\.mesh)", line)
    if match:
        return match.group(1)
    return None

def custom_sort(filename):
    # Extracting the number from the filename
    match = re.search(r'ex(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')  # return a large number for non-matching filenames

def main():
    # Find all cpp files in the current directory
    filepaths = sorted(glob.glob('*.cpp'), key=custom_sort)
    filepaths = [f for f in filepaths if not f.endswith('p.cpp')]
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        basename, _ = os.path.splitext(filename)
        
        # Run make command
        run_command(f"make {basename}")
        
        # Extract and run the sample command from the cpp file
        with open(filepath, 'r') as file:
            for line in file:
                command = extract_sample_run(line)
                if command:
                    command = f"./{command}"
                    run_command(command)
                    break  # Exit the loop once the command is found
            
        if not command:
            print(f"Failed to find the sample command in {filename}")
            continue

        # Rename the output files
        if os.path.exists("sol.gf"):
            os.rename("sol.gf", f"{basename}.gf")
        if os.path.exists("mesh.mesh"):
            os.rename("mesh.mesh", f"{basename}.mesh")
        if os.path.exists("refined.mesh"):
            os.rename("refined.mesh", f"{basename}_refined.mesh")
        
        # Identify mesh name from the command
        mesh_name = command.split("-m")[1].split()[0] if "-m" in command else "mesh.mesh"
        
        # Run the glvis command
        run_command(f"~/glvis-4.2/glvis -m {mesh_name} -g {basename}.gf")

if __name__ == "__main__":
    main()
