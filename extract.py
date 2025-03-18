import argparse
import json
import os
import shutil


def main(root_dir: str, album: str, count: int):
    """
    Extract the top-ranked images from the rankings and copy them to an output directory.
    
    This function copies the top 'count' images from the ranking results of a specific album
    to a designated src_ranked directory within that album.
    
    Args:
        root_dir (str): Root directory containing album folders.
        album (str): The album name to extract top images from.
        count (int): Number of top-ranked images to extract.
    """
    print(f"Extraction started for album '{album}'...")

    # Check if the album directory exists
    album_dir = os.path.join(root_dir, album)
    if not os.path.isdir(album_dir):
        print(f"Error: Album '{album}' not found in '{root_dir}'")
        return

    # Define source and output directories
    src_dir = os.path.join(album_dir, 'src_cropped')
    output_dir = os.path.join(album_dir, 'src_ranked')

    # Create src_ranked directory if it doesn't exist, or clear it if it does
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Check if rankings file exists (updated path to match rank.py changes)
    rankings_path = os.path.join(album_dir, 'rankings.json')
    if not os.path.exists(rankings_path):
        print(f"Error: Rankings file not found at '{rankings_path}'")
        print("Please run the ranking process for this album first.")
        return

    # Load the rankings
    with open(rankings_path, 'r') as f:
        rankings = json.load(f)

    # Ensure we don't try to extract more images than we have in rankings
    if count > len(rankings):
        print(f"Warning: Requested {count} images but only {len(rankings)} are available.")
        count = len(rankings)

    # Copy the top ranked images to the output directory
    successful_copies = 0
    for i in range(count):
        if i >= len(rankings):
            break
            
        image_file, score = rankings[i]
        src_path = os.path.join(src_dir, image_file)
        
        # Get file extension from original file
        _, ext = os.path.splitext(image_file)
        
        # Preserve original filename but prefix with rank
        dst_path = os.path.join(output_dir, f"{i+1:03d}_{image_file}")
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            successful_copies += 1
            print(f"Copied: {image_file} (Score: {score:.2f}) -> {os.path.basename(dst_path)}")
        else:
            print(f"Warning: Could not find source image '{src_path}'")

    print(f"Extraction completed. Copied {successful_copies} of {count} top-ranked images to {output_dir}")
    
    # Create a ranking information file
    info_path = os.path.join(output_dir, "ranking_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Top {count} Images from Album '{album}'\n")
        f.write(f"Ranked on: {os.path.getmtime(rankings_path)}\n\n")
        f.write("Rank, Filename, Score\n")
        for i in range(min(count, len(rankings))):
            image_file, score = rankings[i]
            f.write(f"{i+1}, {image_file}, {score:.2f}\n")
            
    print(f"Created ranking information file at {info_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Top-Ranked Images from an Album")
    parser.add_argument("-r", "--root_dir", type=str, required=True, help="Root directory containing album folders.")
    parser.add_argument("-a", "--album", type=str, required=True, help="Album name to process.")
    parser.add_argument("-c", "--count", type=int, required=True, help="Number of top-ranked images to extract.")
    
    args = parser.parse_args()
    main(args.root_dir, args.album, args.count)
