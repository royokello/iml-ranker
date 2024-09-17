import argparse
import json
import os
import shutil


def main(working_dir: str, count: int):
    """
    """
    print("extraction started ...")

    low_res_input_dir = os.path.join(working_dir, 'cropper', 'output', '256p')
    std_res_input_dir = os.path.join(working_dir, 'cropper', 'output', '512p')

    low_res_output_dir = os.path.join(working_dir, 'ranker', 'output', '256p')
    std_res_output_dir = os.path.join(working_dir, 'ranker', 'output', '512p')

    if os.path.exists(low_res_output_dir):
        shutil.rmtree(low_res_output_dir)
    
    if os.path.exists(std_res_output_dir):
        shutil.rmtree(std_res_output_dir)

    os.makedirs(low_res_output_dir, exist_ok=True)
    os.makedirs(std_res_output_dir, exist_ok=True)

    # Load the rankings
    rankings_path = os.path.join(working_dir, 'ranker', 'rankings.json')
    with open(rankings_path, 'r') as f:
        rankings = json.load(f)

    # Copy the top percentile images to the output directories
    for i in range(count):
        image_file, _ = rankings[i]
        image_base_name = os.path.splitext(image_file)[0]

        low_res_src_path = os.path.join(low_res_input_dir, image_file)
        std_res_src_path = os.path.join(std_res_input_dir, f"{image_base_name}.png")

        low_res_dst_path = os.path.join(low_res_output_dir, f"{i + 1}.png")
        std_res_dst_path = os.path.join(std_res_output_dir, f"{i + 1}.png")

        if os.path.exists(low_res_src_path):
            shutil.copy(low_res_src_path, low_res_dst_path)
        if os.path.exists(std_res_src_path):
            shutil.copy(std_res_src_path, std_res_dst_path)

    print(f"Extracted top {count} of images to {low_res_output_dir} and {std_res_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank Images")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Working Directory in ILM Format.")
    parser.add_argument("-c", "--count", type=int, required=True, help="Top Count.")
    
    args = parser.parse_args()
    main(args.working_dir, args.count)

