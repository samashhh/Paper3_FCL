# Federated Continual Learning (FCL) Project

## Overview
This project implements Federated Continual Learning (FCL) using CIFAR-100 and DWRL datasets. The setup includes client training, model aggregation, and testing with the following methodologies:

1. **Continual Learning (CL):** Training individual client models in sequential data batches.
2. **Federated Learning (FL):** Aggregating models from multiple clients after training on local data.
3. **Federated Continual Learning (FCL):** Combining CL and FL to handle sequential and distributed data.

---

## Project Structure
The project is organized into the following structure:

```
Paper3_FCL/
├── data/                # Data folder for datasets
├── models/              # Saved models
├── notebooks/           # Jupyter notebooks and scripts
│   └── relevant_script3.py  # Main Python script for running experiments
├── src/                 # Source code (if applicable)
├── README.md            # Project documentation
```

---

## Features
- **Dataset Support:**
  - CIFAR-100: Predefined splits for federated learning experiments.
  - DWRL: Custom plastic dataset for recycling experiments.

- **Key Components:**
  - Continual learning simulation with replay strategies.
  - Federated learning using FedAvg.
  - Client and server model training and evaluation.

---

## Requirements
- Python 3.8+
- PyTorch 1.10+
- torchvision
- tqdm

Install the dependencies using:
```bash
pip install -r requirements.txt
```

---

## Usage
### 1. Running Experiments with CIFAR-100
```bash
python3 notebooks/relevant_script3.py --dataset cifar --num_clients 8 --num_classes 20
```

### 2. Running Experiments with DWRL
```bash
python3 notebooks/relevant_script3.py --dataset dwrl --num_clients 4 --num_classes 7
```

### Known Issues
- Running the script with the DWRL dataset results in an error:
  ```
  TypeError: object of type 'NoneType' has no len()
  ```
  Debugging is in progress to resolve this issue. The error is likely related to dataset loading or client splitting.

---

## How to Contribute
1. Fork this repository.
2. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a Pull Request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact
For questions or collaboration, feel free to contact:
- **Name:** Somayeh Shami
- **Email:** somayeh.shami@studio.unibo.it
