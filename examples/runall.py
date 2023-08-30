import os
import glob

# change dir to examples
os.chdir("examples")

def run_command(command):
    os.system(command)

def extract_sample_run(line):
    # Assuming the format is exactly "// Sample runs: command"
    parts = line.split(":")
    if len(parts) > 1:
        return parts[1].strip()
    return None

def main():
    # Find all cpp files in the current directory
    for filepath in sorted(glob.glob('*.cpp')):
        filename = os.path.basename(filepath)
        basename, _ = os.path.splitext(filename)
        
        # Run make command
        run_command(f"make {basename}")
        
        # Extract and run the sample command from the cpp file
        with open(filepath, 'r') as file:
            for line in file:
                if line.startswith("// Sample runs:"):
                    command = extract_sample_run(line)
                    if command:
                        command = f"./{command}"
                        run_command(command)
                    break

        # Rename the output files
        if os.path.exists("sol.gf"):
            os.rename("sol.gf", f"{basename}.gf")
        if os.path.exists("mesh.mesh"):
            os.rename("mesh.mesh", f"{basename}.mesh")
        
        # Identify mesh name from the command
        mesh_name = command.split("-m")[1].split()[0] if "-m" in command else "mesh.mesh"
        
        # Run the glvis command
        run_command(f"~/glvis-4.2/glvis -m ~/mfem/data/{mesh_name} -g mfem/examples/{basename}.gf")

if __name__ == "__main__":
    main()
