from model import Model

if __name__ == "__main__":
    input_image_path = "lane.jpg"  # Ensure this path is correct
    try:
        print("Testing Model...")
        Model.process_and_visualize(input_image_path)
        print("Model tested successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
