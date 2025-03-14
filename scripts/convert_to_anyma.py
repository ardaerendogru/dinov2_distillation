import argparse
import torch
import pickle as pkl

def main(input_ckpt, output_pkl):
    """
    Converts student model weights from a checkpoint file to a pickle file in Anyma format.

    This function loads a PyTorch checkpoint, extracts the student model's backbone weights,
    reformats them to be compatible with the Anyma framework, and saves them as a pickle file.
    It is specifically designed for distilled models where the student model's weights are nested
    under 'state_dict' and 'student.model.model.' keys in the checkpoint.

    Args:
        input_ckpt (str): Path to the input PyTorch checkpoint file (.pth or .ckpt).
                           This checkpoint should contain the distilled model's state_dict,
                           including the student model's weights.
        output_pkl (str): Path to the output pickle file (.pkl) where the converted weights
                           will be saved. The output file will be in a format compatible with
                           the Anyma framework for model initialization.
    """
    w = torch.load(input_ckpt)
    w_student = {k: v for k, v in w['state_dict'].items() if 'student' in k and 'scalekd' not in k and 'feature_matchers' not in k}
    
    weights = {}

    for k, v in w_student.items():
        new_k = k.replace('student.model.model.', 'backbone.')
        weights[new_k] = v.detach().cpu().numpy()

    # Save in Detectron2 compatible format with metadata
    res = {
        "model": weights,
        "__author__": "dinov2_distilled",
        "matching_heuristics": True
    }

    with open(output_pkl, "wb") as f:
        pkl.dump(res, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert student model weights to Anyma format.")
    parser.add_argument("input_ckpt", type=str, help="Path to the input checkpoint file.")
    parser.add_argument("output_pkl", type=str, help="Path to the output pickle file.")
    args = parser.parse_args()

    """
    Entry point for the script to convert student model weights to Anyma format.

    Parses command-line arguments for input checkpoint path and output pickle file path,
    and then calls the main conversion function.

    Usage:
        python convert_to_anyma.py <input_checkpoint_path> <output_pickle_path>

    Example:
        python convert_to_anyma.py student_model.ckpt anyma_weights.pkl
    """
    main(args.input_ckpt, args.output_pkl)