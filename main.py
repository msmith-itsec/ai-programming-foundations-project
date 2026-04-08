"""
Command-line menu for the Prompt Injection Detection project.
"""

from __future__ import annotations

from predict import predict_text
from train_model import train_pipeline


def main() -> None:
    while True:
        print("\n=== Prompt Injection Detection Using Machine Learning ===")
        print("1. Train model")
        print("2. Run prediction demo")
        print("3. Exit")

        choice = input("Select an option (1-3): ").strip()

        if choice == "1":
            metrics = train_pipeline()
            print("\nTraining finished.")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
        elif choice == "2":
            prompt = input("Enter a prompt to classify: ").strip()
            try:
                print(f"Prediction: {predict_text(prompt)}")
            except FileNotFoundError as exc:
                print(exc)
        elif choice == "3":
            print("Goodbye.")
            break
        else:
            print("Invalid option. Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
