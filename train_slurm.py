"""Sets up config and data directories for a batch of
experiments, and creates the necessary `sbatch` file."""
import argparse
import os.path
import subprocess

from hi_c import setup_experiments, timestamp

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("config_files", type=str, nargs="*",
                        help="provide one or more experiment config files")

    parser.add_argument("-o", "--output-path", type=str, default="./results/debug",
                        help="directory in which we should save results (will be mounted in each Singularity container)")

    parser.add_argument("--num-seeds", type=int,
                        help="the number of random seeds to run, overrides values from the config file")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="a list of random seeds to run, overrides values from the config file")

    parser.add_argument("--flush-secs", type=int, default=200,
                        help="number of seconds after which we should flush the training longs")

    parser.add_argument("-i", "--image", type=str, default="./singularity_image.sif",
                        help="singularity image in which to run experiments")

    parser.add_argument("-p", "--partition", type=str,
                        help="name of SLURM partition to use")
    parser.add_argument("-q", "--qos", type=str,
                        help="SLURM QoS to use")
    parser.add_argument("-t", "--time", type=str, default="1:00:00",
                        help="SLURM time limit per job")
    parser.add_argument("-c", "--cpus-per-task", type=int, default=1,
                        help="CPUs per SLURM task")
    parser.add_argument("-g", "--gpus-per-task", type=float, default=0,
                        help="GPUs per SLURM task")
    parser.add_argument("-m", "--mem-per-cpu", type=str, default="512M",
                        help="memory per SLURM CPU")
    parser.add_argument("--slurm-output", type=str, default="./slurm_output/",
                        help="directory to store SLURM output")

    parser.add_argument("--setup", action="store_true",
                        help="setup experiments and create sbatch script, but do not launch jobs")

    return parser.parse_known_args()


def get_filename(base_name, index_digits=3):
    base_name = base_name + "_" + timestamp()
    name = base_name + ".sh"

    idx = 0
    while os.path.exists(name):
        idx += 1
        name = base_name + "_" + str(idx).zfill(index_digits) + ".sh"

    return name


if __name__ == '__main__':
    args, unknown = parse_args()

    # Initialize experiment directories
    exp_paths = setup_experiments(args.config_files, 
                              args.output_path, 
                              num_seeds=args.num_seeds,
                              seeds=args.seeds, 
                              arguments=unknown)

    # Initialize run command
    run_command = " ".join([
        "singularity", 
        "exec",
        "--bind",
        f"{args.output_path}:/mnt/output",  # NOTE: The first on should be the local path, the second the path within the container
        args.image,
        "python3", 
        "slurm_run.py",
        "--flush-secs",
        str(args.flush_secs)
    ])

    if args.gpus_per_task > 0:
        run_command += " --gpu"

    # Initialize sbatch header
    slurm_header = [
        "#!/bin/sh\n",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --qos={args.qos}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --gpus-per-task={args.gpus_per_task}",
        f"#SBATCH --mem-per-cpu={args.mem_per_cpu}",
    ]

    # Create slurm output directory if needed
    os.makedirs(args.slurm_output, exist_ok=True)

    # Create and launch sbatch files
    for name, paths in exp_paths.items():
        sbatch_path = get_filename(name)
        root_output = os.path.join(args.slurm_output,  name + r"_%j")
        task_output = os.path.join(args.slurm_output,  name + r"_%J")

        with open(sbatch_path, "w") as f:
            f.writelines(slurm_header)
            f.write(f"#SBATCH --job-name={name}\n")            
            f.write(f"#SBATCH --output={root_output}\n\n")

            for path in paths:
                f.write(f"srun -n1 --output={task_output} {run_command} ")
                f.write(path)
                f.write(" &\n")
            
            f.write("wait")
        
        if not args.setup:
            subprocess.run(f"sbatch {sbatch_path}")
