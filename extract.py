import argparse
import csv
import os
import shutil

def get_latest_stage(project_path: str) -> int | None:
    """Finds the latest stage number in the project directory."""
    latest_stage = -1
    if not os.path.isdir(project_path):
        print(f"Warning: Project path '{project_path}' not found or not a directory when searching for stages.")
        return None
    for item in os.listdir(project_path):
        if os.path.isdir(os.path.join(project_path, item)) and item.startswith("stage_"):
            try:
                stage_num_str = item.split("_")[1]
                stage_num = int(stage_num_str)
                if stage_num > latest_stage:
                    latest_stage = stage_num
            except (IndexError, ValueError):
                # Not a valid stage folder name like stage_X
                continue
    return latest_stage if latest_stage != -1 else None

def main(project_path: str, count: int, stage: int | None = None):
    """
    Extract the top-ranked images from a specific project stage and copy them to an output directory.
    
    Args:
        project_path (str): Path to the project directory.
        count (int): Number of top-ranked images to extract.
        stage (int, optional): The stage number. If None, the latest stage is used.
    """
    if not os.path.isdir(project_path):
        print(f"Error: Project directory '{project_path}' not found.")
        return

    if stage is None:
        print("Stage not provided, attempting to find the latest stage...")
        stage = get_latest_stage(project_path)
        if stage is None:
            print(f"Error: No stages found in project '{project_path}'. Ensure stage folders are named 'stage_X'.")
            return
        print(f"Using latest stage: stage_{stage}")
    
    stage_folder_name = f"stage_{stage}"
    # Source directory for images is the stage folder itself
    src_dir = os.path.join(project_path, stage_folder_name)

    if not os.path.isdir(src_dir):
        print(f"Error: Stage directory '{src_dir}' not found in project '{project_path}'.")
        return
        
    print(f"Extraction started for project '{os.path.basename(project_path)}', source stage '{stage_folder_name}'...")

    # Define the next stage directory for output
    next_stage_number = stage + 1
    output_dir = os.path.join(project_path, f"stage_{next_stage_number}")
    print(f"Outputting selected images to new stage directory: '{os.path.basename(output_dir)}' at '{output_dir}'")

    # Create/clear output directory (new stage folder)
    # This directory will store the extracted images and the extraction_info.txt
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Rankings are stored at f"stage_{stage}_rankings.csv" in the project directory
    rankings_filename = f"stage_{stage}_rankings.csv"
    rankings_path = os.path.join(project_path, rankings_filename)

    # Load the rankings from CSV
    rankings = []
    try:
        with open(rankings_path, 'r', newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader) # Skip header row, e.g., "image_filename,score"
            except StopIteration:
                print(f"Error: Rankings file '{rankings_path}' is empty.")
                return
            # Optional: Validate header if needed
            # if header != ['image_filename', 'score']:
            #     print(f"Warning: CSV header in '{rankings_path}' is {header}. Expected ['image_filename', 'score'].")

            for row_num, row in enumerate(reader, 1):
                if len(row) == 2:
                    image_file, score_str = row
                    try:
                        rankings.append((image_file, float(score_str)))
                    except ValueError:
                        print(f"Warning: Could not parse score '{score_str}' for image '{image_file}' in '{rankings_path}' (row {row_num+1}). Skipping.")
                else:
                    print(f"Warning: Skipping malformed row {row_num+1} in '{rankings_path}': {row}. Expected 2 columns.")
        # Assuming rankings in CSV are already sorted by score in descending order.
        # If not, they need to be sorted here, e.g.:
        # rankings.sort(key=lambda item: item[1], reverse=True)

    except FileNotFoundError:
        print(f"Error: Rankings file not found at '{rankings_path}'")
        print("Please run the ranking process for this stage first.")
        return
    except Exception as e:
        print(f"Error reading or parsing rankings file '{rankings_path}': {e}")
        return

    if not rankings:
        print(f"No rankings loaded from '{rankings_path}'. Nothing to extract.")
        return

    # Ensure we don't try to extract more images than we have in rankings
    actual_count = min(count, len(rankings))
    if count > len(rankings):
        print(f"Warning: Requested {count} images but only {len(rankings)} are available in rankings. Extracting {actual_count}.")

    # Copy the top ranked images to the output directory
    successful_copies = 0
    for i in range(actual_count):
        image_file, score = rankings[i]
        # Images are directly in the stage folder (src_dir)
        src_path = os.path.join(src_dir, image_file)
        
        # Preserve original filename but prefix with rank
        dst_path = os.path.join(output_dir, f"{i+1:03d}_{image_file}")
        
        if os.path.exists(src_path):
            try:
                shutil.copy(src_path, dst_path)
                successful_copies += 1
                # print(f"Copied: {image_file} (Score: {score:.2f}) -> {os.path.basename(dst_path)}")
            except Exception as e:
                print(f"Error copying '{src_path}' to '{dst_path}': {e}")
        else:
            print(f"Warning: Could not find source image '{src_path}' for ranked item '{image_file}'.")

    print(f"Extraction completed. Copied {successful_copies} of {actual_count} top-ranked images to {output_dir}")
    
    # Create an extraction information file in the new stage directory
    info_path = os.path.join(output_dir, "extraction_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Top {actual_count} Images extracted from Project '{os.path.basename(project_path)}', Source Stage '{stage_folder_name}'\n")
        if os.path.exists(rankings_path):
             f.write(f"Rankings based on: {rankings_filename} (Last modified: {os.path.getmtime(rankings_path)})\n\n")
        else:
            # This case should ideally not be reached if rankings loaded successfully earlier
            f.write(f"Rankings based on: {rankings_filename} (File not found at time of info creation)\n\n")
        f.write("Rank, Filename, Score\n")
        for i in range(actual_count):
            image_file, score = rankings[i]
            f.write(f"{i+1}, {image_file}, {score:.2f}\n")
            
    print(f"Created extraction information file at {info_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Top-Ranked Images from a Project Stage")
    parser.add_argument("--project", type=str, required=True, help="Path to the project directory.")
    parser.add_argument("--stage", type=int, required=False, default=None, help="Stage number (optional, defaults to the latest stage found in project directory e.g. stage_X).")
    parser.add_argument("--count", type=int, required=True, help="Number of top-ranked images to extract.")
    
    args = parser.parse_args()
    main(project_path=args.project, stage=args.stage, count=args.count)
