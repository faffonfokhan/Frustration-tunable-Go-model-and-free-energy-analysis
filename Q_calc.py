import MDAnalysis as mda
import numpy as np
import math
import argparse
import os
from pandas import DataFrame
from tqdm import tqdm

def extract_pairs_from_top(top_file, output_pairs_file="pairs.txt"):
    pairs = []
    copying = False

    with open(top_file, 'r') as infile, open(output_pairs_file, 'w') as outfile:
        for line in infile:
            if line.strip().startswith("[ pairs ]"):
                copying = True
                continue
            if copying and line.strip().startswith("["):
                break
            if copying and not line.strip().startswith(";") and line.strip():
                outfile.write(line)
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        a1 = int(parts[0])
                        a2 = int(parts[1])
                        pairs.append((a1, a2))
                    except Exception:
                        continue
    print(f"Pairs extracted to {output_pairs_file}")
    return pairs

def calculate_native_distances(pdb_file, pairs, unit='nm'):
    coords = []
    with open(pdb_file, 'r') as pdb:
        for line in pdb:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Gromacs PDB is nm, but many PDBs are in Ångstrom
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                coords.append([x, y, z])
    coords = np.array(coords)

    # Heuristic for units: If largest distance > 50, likely Å, else nm
    if np.max(coords) > 50:
        print("Warning: PDB coordinates in Ångstrom. Converting to nm.")
        coords /= 10.0
        unit = 'Å'
    else:
        unit = 'nm'

    dists = []
    for a1, a2 in pairs:
        d = np.linalg.norm(coords[a1-1] - coords[a2-1])  # 1-based to 0-based
        dists.append(d)
    return np.array(dists), unit

def calculate_q(top_file, xtc_file, pdb_file, beta=5.0, lambda_param=1.2):
    pairs = extract_pairs_from_top(top_file)
    if len(pairs) == 0:
        raise RuntimeError("No pairs found in topology.")
    native_dists, unit = calculate_native_distances(pdb_file, pairs)
    u = mda.Universe(pdb_file, xtc_file)

    n_contacts = len(pairs)
    Q_vals = []
    contact_indices = np.array([(a1-1, a2-1) for a1, a2 in pairs])  # 0-based
    print(f"Calculating Q for {len(u.trajectory)} frames, {n_contacts} contacts...")
    for ts in tqdm(u.trajectory):
        positions = u.atoms.positions  # shape (n_atoms, 3)
        dists = np.linalg.norm(positions[contact_indices[:, 0]] - positions[contact_indices[:, 1]], axis=1)
        exp_arg = beta * (dists - lambda_param * native_dists)
        exp_arg = np.clip(exp_arg, -700, 700)  # Prevent overflow
        q_i = 1.0 / (1.0 + np.exp(exp_arg))
        Q_frame = np.mean(q_i)
        Q_vals.append(Q_frame)
    output_file = "Q_values.csv"
    DataFrame(Q_vals).to_csv(output_file, index=False)
    print(f"Q values calculated and saved to {output_file} ({len(Q_vals)} frames)")
    return Q_vals

def main():
    parser = argparse.ArgumentParser(description='Calculate Q values from trajectory.')
    parser.add_argument('-top', required=True, help='Topology file (.top)')
    parser.add_argument('-xtc', required=True, help='Trajectory file (.xtc)')
    parser.add_argument('-pdb', required=True, help='PDB reference file (.pdb)')
    parser.add_argument('--beta', type=float, default=5.0, help='Beta parameter for switching function (default: 5.0)')
    parser.add_argument('--lambda_param', type=float, default=1.2, help='Lambda parameter for contact distance (default: 1.2)')
    args = parser.parse_args()

    for file_path, file_type in [(args.top, "Topology"), (args.xtc, "Trajectory"), (args.pdb, "PDB")]:
        if not os.path.exists(file_path):
            print(f"Error: {file_type} file '{file_path}' does not exist.")
            return

    calculate_q(args.top, args.xtc, args.pdb, beta=args.beta, lambda_param=args.lambda_param)

if __name__ == "__main__":
    main()
