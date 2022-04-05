# BuchiAutomata_for_GNN

Code accompanying the paper "Analyzing Buchi Automata With Graph Neural Networks" (link soon).

# "BA_forGNN_Generator.py"
This python file takes care of the dataset implementation discussed in the paper.
To generate a balanced dataset and save it to your computer, it suffices to call the function

generate_dataset(
- string denoting the desired property (checked for the automaton classification label) - "empty", "min1b" or "infb"
- int denoting the size of the dataset (number of automata)
- dict containing the generation parameters. Use predefined "paper_parameters" to follow paper description

)

# "BA_baseGNN.py"
This python file uses the created datasets to train basic neural networks and save the classifying accuracies
To train the GNN, call the following function:

create_and_train_nn(
- EPOCHS: int denoting the number of training epochs
- BATCH_SIZE: int denoting the size of the dataset batches
- HIDDEN_CHANNELS: int denoting the number of hidden channels of the GNN
- trainsrc: the name of the datasetfile in the datasetfolder used for training
- testsrc: the name of the datasetfile in the datasetfolder used for testing

)



