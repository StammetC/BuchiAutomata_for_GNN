import torch
import os
import numpy as np
import random
from itertools import chain, combinations
from typing import Tuple

import torch_geometric.data
from PySimpleAutomata import automata_IO
from torch_geometric.data import Data
from string import ascii_lowercase

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

if not os.path.exists("graphicaldata"):
    os.makedirs("graphicaldata")
dataset_folder = "datasets/test"
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)


def generate_nbw_er_eachsymbol(nmin: int, nmax: int, pmin: float,
                               pmax: float, paccmin: float, paccmax: float,
                               s: int, nadd: int, featinit: str = "half") -> Tuple[
                               torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """
    Generates one NBW with the given parameters.
    Uses the Erdös-Renyi model for each possible transition (i.e. for each poss. symbol in Sigma)
    to randomly generate automaton
    Return Tensors are formatted to be given as parameters to torch_geometric.data.Data constructor.

    :param nmin: lower bound of possible number of nodes
    :param nmax: upper bound of possible number of nodes
    :param pmin: lower bound of probability of transition existence
    :param pmax: upper bound of probability of transition existence
    :param paccmin: lower bound of probability of node being accepting
    :param paccmax: upper bound of probability of node being accepting
    :param s: alphabet size
    :param nadd: number of additional node feature vector elements
    :param featinit: how to init add elements: "random", "half" (default) or "zero"
    :return: edge_index, edge_attr and x for Pytorch.data element creation
    """
    n = random.randint(nmin, nmax)
    p = (random.randint(100 * pmin, 100 * pmax)) / 100
    acc_p = (random.randint(100 * paccmin, 100 * paccmax)) / 100
    # Create alphabet Sigma with one-hot encoded characters
    sigma = []
    for i in range(s):
        char = np.zeros(s, dtype=np.int8)
        char[i] = 1
        sigma.append(list(char))
    edge_in = []
    edge_out = []
    edge_attr = []
    # rolls a dice for each src, dest state and each character
    # if roll is above probability p, transition is added to automaton
    for src in range(n):
        for dest in range(n):
            for c in sigma:
                diceroll = random.random()
                if diceroll < p:
                    # We have an edge, add to tensors
                    edge_in.append(src)
                    edge_out.append(dest)
                    edge_attr.append(c)
    # encoding of transitions as arguments for torch.Data constructor
    edge_index = torch.tensor([edge_in, edge_out], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr)

    # create the node feature vectors
    x = []
    acc_states = []
    # creates feature_vector for each node
    for i in range(n):
        # initialization as "half", "random" or other (all zeros)
        if featinit == "random":
            feature_vector = np.random.rand(2 + nadd)
        else:
            feature_vector = np.zeros(2 + nadd, dtype=float)
            if featinit == "half":
                feature_vector += 0.5
        # first node is initial (by convention)
        if i == 0:
            feature_vector[0] = 1
        else:
            feature_vector[0] = 0
        # diceroll to determine acceptance of state
        diceroll = random.random()
        if diceroll < acc_p:
            feature_vector[1] = 1
            acc_states.append(i)
        else:
            feature_vector[1] = 0
        # adds feature vector to torch.Data input Tensor x
        x.append(feature_vector)
    x = torch.tensor(x, dtype=torch.float32)
    return edge_index, edge_attr, x, acc_states


def generate_nbw_er_adjmatrix(nmin: int, nmax: int, pmin: float,
                              pmax: float, paccmin: float, paccmax: float,
                              s: int, nadd: int, featinit: str = "half") -> Tuple[
                              torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates one NBW with the given parameters.
    Uses the Erdös-Renyi model for the adjacency matrix,
    then chooses label for each edge from symbol power-set to randomly generate automaton
    Return Tensors are formatted to be given as parameters to torch_geometric.data.Data constructor.

    :param nmin: lower bound of possible number of nodes
    :param nmax: upper bound of possible number of nodes
    :param pmin: lower bound of probability of transition existence
    :param pmax: upper bound of probability of transition existence
    :param paccmin: lower bound of probability of node being accepting
    :param paccmax: upper bound of probability of node being accepting
    :param s: alphabet size
    :param nadd: number of additional node feature vector elements
    :param featinit: how to init add elements: "random", "half" (default) or "zero"
    :return: edge_index, edge_attr and x for Pytorch.data element creation
    """
    n = random.randint(nmin, nmax)
    p = (random.randint(100 * pmin, 100 * pmax)) / 100
    acc_p = (random.randint(100 * paccmin, 100 * paccmax)) / 100
    # Create alphabet Sigma with one-hot encoded characters
    sigma = []
    for i in range(s):
        char = np.zeros(s, dtype=np.int8)
        char[i] = 1
        sigma.append(list(char))
    edge_in = []
    edge_out = []
    edge_attr = []
    # rolls a dice for each src, dest state
    # if roll is above probability p, transition is added to automaton
    for src in range(n):
        for dest in range(n):
            diceroll = random.random()
            if diceroll < p:
                # determines what symbols are read by the transition
                # creating power-set of Sigma (excluding empty set)
                possible_transitions = chain.from_iterable(
                    combinations(list(sigma), r) for r in range(1, len(sigma) + 1))
                pt = [list(i) for i in possible_transitions]
                # adds transition labels for each one in randomly chosen element in power-set
                for t in random.choice(pt):
                    edge_in.append(src)
                    edge_out.append(dest)
                    edge_attr.append(t)
    # encoding of transitions as arguments for torch.Data constructor
    edge_index = torch.tensor([edge_in, edge_out], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr)

    # Now let's create the node feature vectors
    x = []
    acc_states = []
    # creates feature_vector for each node
    for i in range(n):
        # initialization as "half", "random" or other (all zeros)
        if featinit == "random":
            feature_vector = np.random.rand(2 + nadd)
        else:
            feature_vector = np.zeros(2 + nadd, dtype=float)
            if featinit == "half":
                feature_vector += 0.5
        # first node is initial (by convention)
        if i == 0:
            feature_vector[0] = 1
        else:
            feature_vector[0] = 0
        # diceroll to determine acceptance of state
        diceroll = random.random()
        if diceroll < acc_p:
            feature_vector[1] = 1
            acc_states.append(i)
        else:
            feature_vector[1] = 0
        # adds feature vector to torch.Data input Tensor x
        x.append(feature_vector)
    x = torch.tensor(x, dtype=torch.float32)
    return edge_index, edge_attr, x


def save_automata_from_data(data: torch_geometric.data.Data, filename: str, draw: bool = False) -> None:
    """
    Takes as input a dataelement and a foldername and creates a .dot text representation of
    the given automaton and optionally (boolean 'draw' parameter) also creates a .svg graphical
    representation of the given automaton.
    WARNING: For large automata, this .svg may not be very useful due to readability issues

    :param data: The dataelement to be transformed into a .dot file
    :param filename: The folder where to store the .dot file
    :param draw: If true, also adds a graphical .svg representation
    :return: None - adds a file to the given folder
    """
    src = r"./graphicaldata/" + filename + ".txt"
    file = open(src, "w")
    file.write("digraph{\n")
    file.write("fake [style=invisible]\n")
    number_of_nodes = len(data.x)
    number_of_transitions = len(data.edge_attr)
    if not (number_of_transitions == 0):
        number_of_characters = len(data.edge_attr[0])
    else:
        number_of_characters = 2
    state_names = []
    char_names = []
    # create all the node names
    for i in range(number_of_nodes):
        name = "q" + str(i)
        state_names.append(name)
    # create all the character names
    for i in range(number_of_characters):
        name = ascii_lowercase[i]
        char_names.append(name)
    # check for initial state and write to file
    for i in range(number_of_nodes):
        if data.x[i][0] == 1:
            file.write("fake -> ")
            file.write(state_names[i])
            file.write(" [style=bold]\n")
            initial_index = i
    # write all the state names and features to file
    for i in range(number_of_nodes):
        # state is acc and init
        if data.x[i][0] == 1 and data.x[i][1] == 1:
            file.write(state_names[i])
            file.write(" [root=true, shape=doublecircle]\n")
        # state is init
        elif data.x[i][0] == 1:
            file.write(state_names[i])
            file.write(" [root=true]\n")
        # state is acc
        elif data.x[i][1] == 1:
            file.write(state_names[i])
            file.write(" [shape=doublecircle]\n")
        # other states
        else:
            file.write(state_names[i])
            file.write("\n")
    transition_infos = []
    for t in range(number_of_transitions):
        src_index = data.edge_index[0][t]
        dest_index = data.edge_index[1][t]
        chara_index = np.where(data.edge_attr[t].numpy() == 1)[0][0]
        transition_infos.append([src_index, dest_index, [chara_index]])
    to_remove = []
    for i in range(len(transition_infos) - 1):
        if (transition_infos[i][0] == transition_infos[i + 1][0]) and (
                transition_infos[i][1] == transition_infos[i + 1][1]):
            gone = transition_infos[i]
            to_remove.append(i)
            for g in gone[2]:
                transition_infos[i + 1][2].append(g)
    for index in sorted(to_remove, reverse=True):
        del transition_infos[index]
    for t in transition_infos:
        t[2] = sorted(t[2])
        file.write(state_names[t[0]])
        file.write("->")
        file.write(state_names[t[1]])
        file.write(" [label=\"")
        if len(t[2]) == 1:
            file.write(char_names[t[2][0]])
            file.write("\"]\n")
        else:
            file.write(char_names[t[2][0]])
            for i in range(len(t[2]) - 1):
                file.write(",")
                file.write(char_names[t[2][i + 1]])
            file.write("\"]\n")
    file.write("}")
    file.close()
    if draw:
        draw_graph_from_dot(src)


def save_automata_from_dataset(data: list, foldername: str, draw: bool = False) -> None:
    """
    Takes as input a (slice of a) dataset of automata and a foldername and creates a .dot text representation of
    each given automaton and optionally (boolean 'draw' parameter) also creates a .svg graphical
    representation of the given automata. \n
    WARNING: For large datasets, this will be very time-consuming. Rather use for debugging/testing on small datasets \n
    WARNING: For sets of large automata, this .svg may not be very useful due to readability issues

    :param data: The set of dataelements to be transformed into a .dot file each
    :param foldername: The folder where to store the .dot file
    :param draw: If true, also adds a graphical .svg representation for each automaton
    :return: None - adds the .dot (and optional .svg) files to the given folder
    """
    count = 0
    if not os.path.exists("graphicaldata/" + foldername):
        os.makedirs("graphicaldata/" + foldername)
    for d in data:
        src = foldername + "/" + str(count) + "_" + str(d.y) + "_" + str(d.cyclelen)
        save_automata_from_data(d, src, draw)
        count += 1


def draw_graph_from_dot(path: os.path) -> None:
    """
    Takes as input a path to a .dot text representation of an automaton and adds a .svg graphical
    representation of the same automaton to the same folder.

    :param path: path of a .dot text representation of an automaton
    :return: None - adds a graphical .svg representation of the given automaton (as a .dot file)
    """
    folder, file = os.path.split(path)
    nfa_from_file = automata_IO.nfa_dot_importer(path)
    automata_IO.nfa_to_dot(nfa_from_file, file, folder)
    os.remove(path)


def get_minimal_distance(edge_in: list, edge_out: list, s1: int, s2: int) -> int:
    """
    Takes as input two lists describing the edge relations of an automaton (using the conventions used
    by the torch_geometric.data.Data class and returns the minimal distance between
    the two given nodes s1 and s2 using a breath first search algorithm.
    -1 is returned as a distance if s1 and s2 are not connected.

    :param edge_in: first list from 'edge_index' tensor from the torch_geometric.data.Data class
    :param edge_out: second list from 'edge_index' tensor from the torch_geometric.data.Data class
    :param s1: integer number of a node in edge list (start of searched path)
    :param s2: integer number of a node in edge list (goal of searched path)
    :return: integer denoting the minimal distance between nodes s1 and s2 ('-1' if s1 and s2 are not connected)
    """
    dist = 1
    done = []
    # dist_hop denotes the list of all nodes that are 'dist' away from s1
    dist_hop = [s1]
    # distplusone_hop is the list of all nodes that are 'dist'+1 away from s1
    distplusone_hop = []
    # a safety measure while loop - distance cannot be longer than the total number of edges
    while dist <= len(edge_in):
        for s in dist_hop:
            done.append(s)
            # computes the list of all direct successors of s
            occurrences = lambda s, lst: (i for i, e in enumerate(lst) if e == s)
            all_succ = set([edge_out[i] for i in occurrences(s, edge_in)])
            # check for each successor if it 1) is the goal state s2 or 2) has already been treated by the algorithm
            for succ in all_succ:
                if succ == s2:
                    return dist
                elif not (succ in done) and not (succ in distplusone_hop):
                    distplusone_hop.append(succ)
        # jumping one step forward - next run of the while loop, all treated states' distance increases by 1
        dist += 1
        dist_hop = distplusone_hop
        distplusone_hop = []
    return -1


def get_minimal_distances_to_acc(edge_in: list, edge_out: list, acc: list) -> Tuple[int, int]:
    """
    Takes as input two lists describing the edge relations of an automaton (using the conventions used
    by the torch_geometric.data.Data class and returns both the minimal distance from the initial state
    to an accepting state as well as the minimal distance from any accepting state to itself.

    :param edge_in: first list from 'edge_index' tensor from the torch_geometric.data.Data class
    :param edge_out: second list from 'edge_index' tensor from the torch_geometric.data.Data class
    :param acc: list of accepting states of given automaton
    :return: integer denoting the minimal distance between initial and any accepting state (-1 if not (self-)reachable)
    """
    if 0 in acc:
        to_acc = 0
    else:
        to_acc = len(edge_in) + 1
    cycle_acc = len(edge_in) + 1
    for a in acc:
        x = get_minimal_distance(edge_in, edge_out, 0, a)
        y = get_minimal_distance(edge_in, edge_out, a, a)
        if 0 <= x < to_acc and not (y == -1):
            to_acc = x
        if 0 <= y < cycle_acc and not (x == -1):
            cycle_acc = y
    if to_acc == len(edge_in) + 1:
        to_acc = -1
    if cycle_acc == len(edge_in) + 1:
        cycle_acc = -1
    return to_acc, cycle_acc


def non_emptyness_check(edge_in: list, edge_out: list, acc: list, start: int = 0,
                        mode: str = 'from_start') -> Tuple[bool, str]:
    """
    Takes as input two lists describing the edge relations of an automaton (using the conventions used
    by the torch_geometric.data.Data class and the set of accepting states and returns whether the given
    automaton structure is empty or not, i.e. if an accepting state can be reached from the initial state.
    The parameters start and mode are used for a recursive call of the function (do not give as input outside
    of this function). \n
    The string describing the type of emptyness is as follows: \n
    - "non-empty" if the automaton is non empty, i.e. it accepts at least one omega-word \n
    If the automaton is empty, i.e. no omega-word is accepted, the following types are distinguished: \n
    - "no_acc_state": The automaton does not contain an accepting state \n
    - "no_acc_reached": No accepting state is reachable \n
    - "no_acc_selfreached": No reachable accepting state is self-reachable

    :param edge_in: first list from 'edge_index' tensor from the torch_geometric.data.Data class
    :param edge_out: second list from 'edge_index' tensor from the torch_geometric.data.Data class
    :param acc: set of integers denoting the accepting states of the automaton
    :param start: variable needed for recursive function call
    :param mode: variable needed for recursive function call
    :return: boolean denoting non-emptyness of the given automaton and a string describing the type of emptyness
    """
    if len(acc) == 0:
        return False, 'no_acc_state'
    todo = [start]
    reachable_acc = []
    done = []
    acc_has_been_reached = False
    while not (len(todo) == 0):
        active_state = todo.pop(0)
        done.append(active_state)
        # get list of all successors of active_state
        occurrences = lambda s, lst: (i for i, e in enumerate(lst) if e == s)
        all_succ = set([edge_out[i] for i in occurrences(active_state, edge_in)])
        for s in all_succ:
            if not (s in done):
                todo.append(s)
            if (s in acc) and not (s in reachable_acc):
                reachable_acc.append(s)
    if (mode == 'from_acc') and (len(reachable_acc) > 0):
        return True, 'non_empty'
    for s in reachable_acc:
        acc_has_been_reached = True
        dist_from_start = get_minimal_distance(edge_in, edge_out, start, s)
        can_s_reach_itself, type = non_emptyness_check(edge_in, edge_out, [s], start=s, mode='from_acc')
        if can_s_reach_itself:
            return True, 'non_empty'
    if not acc_has_been_reached:
        return False, 'no_acc_reached'
    else:
        return False, 'no_acc_selfreached'


def accepting_min1b_check(edge_in: list, edge_out: list, edge_attr: torch.Tensor,
                          acc: list, start: int = 0, mode: str = 'from_start',
                          b_was_read: bool = False) -> bool:
    """
    Takes as input an automaton structure and the transition labels (encoded as in
    torch_geometric.data.Data class) an outputs whether the given automaton accepts an
    omega-word containing at least one 'b'

    :param edge_in: first list from 'edge_index' tensor from the torch_geometric.data.Data class
    :param edge_out: second list from 'edge_index' tensor from the torch_geometric.data.Data class
    :param edge_attr: list of edge attributes (i.e. the symbol being read)
    :param acc: list of accepting states
    :param start: what node to start the accepting state search
    :param mode: describing if in start or recursive call of the function (denoting starting at an already found accepting state)
    :param b_was_read: boolean describing if a b has been read in prior function call
    :return: True if given automaton accepts at least one omega-word containing at least one 'b'
    """
    if len(acc) == 0:
        return False
    todo = [[start, b_was_read]]
    reachable_acc = []
    done = []
    while not (len(todo) == 0):
        active = todo.pop(0)
        active_state = active[0]
        b_was_read = active[1]
        done.append([active_state, b_was_read])
        # get list of all successors of active_state after reading active_char
        occurrences = lambda s, lst: (i for i, e in enumerate(lst) if e == s)
        all_succ = set([edge_out[i] for i in occurrences(active_state, edge_in)])
        if not b_was_read:
            for i in occurrences(active_state, edge_in):
                # check if 2nd element of one-hot encoding is a 1 (i.e. a b)
                if 1 == edge_attr[i].tolist()[1]:
                    b_was_read = True
        for s in all_succ:
            if not ([s, b_was_read] in done):
                todo.append([s, b_was_read])
            if (s in acc) and not ([s, b_was_read] in reachable_acc):
                if not mode == 'from_acc':
                    reachable_acc.append([s, b_was_read])
                elif b_was_read:
                    reachable_acc.append([s, b_was_read])
    if (mode == 'from_acc') and (len(reachable_acc) > 0):
        return True
    for s in reachable_acc:
        can_s_reach_itself = accepting_min1b_check(edge_in, edge_out, edge_attr, [s[0]], start=s[0], mode='from_acc', b_was_read = s[1])
        if can_s_reach_itself:
            return True
    return False


def accepting_infb_check(edge_in: list, edge_out: list, edge_attr: torch.Tensor,
                         acc: list, start: int = 0, mode: str = 'from_start',
                         b_was_read: bool = False) -> bool:
    """
    Takes as input an automaton structure and the transition labels (encoded as in
    torch_geometric.data.Data class) an outputs whether the given automaton accepts an
    omega-word containing infinitely many 'b'

    :param edge_in: first list from 'edge_index' tensor from the torch_geometric.data.Data class
    :param edge_out: second list from 'edge_index' tensor from the torch_geometric.data.Data class
    :param edge_attr: list of edge attributes (i.e. the symbol being read)
    :param acc: list of accepting states
    :param start: what node to start the accepting state search
    :param mode: describing if in start or recursive call of the function (denoting starting at an already found accepting state)
    :param b_was_read: boolean describing if a b has been read in prior function call
    :return: True if given automaton accepts at least one omega-word containing infinitely many 'b'
    """
    if len(acc) == 0:
        return False
    todo = [[start, b_was_read]]
    reachable_acc = []
    done = []
    while not (len(todo) == 0):
        active = todo.pop(0)
        active_state = active[0]
        b_was_read = active[1]
        done.append([active_state, b_was_read])
        # get list of all successors of active_state after reading active_char
        occurrences = lambda s, lst: (i for i, e in enumerate(lst) if e == s)
        all_succ = set([edge_out[i] for i in occurrences(active_state, edge_in)])
        if not b_was_read:
            for i in occurrences(active_state, edge_in):
                if [0,1] == edge_attr[i].tolist():
                    b_was_read = True
        for s in all_succ:
            if not ([s, b_was_read] in done):
                todo.append([s, b_was_read])
            if (s in acc) and not ([s, b_was_read] in reachable_acc):
                if not mode == 'from_acc':
                    reachable_acc.append([s, b_was_read])
                elif b_was_read:
                    reachable_acc.append([s, b_was_read])
    if (mode == 'from_acc') and (len(reachable_acc) > 0):
        return True
    for s in reachable_acc:
        can_s_reach_itself = accepting_infb_check(edge_in, edge_out, edge_attr, [s[0]], start=s[0], mode='from_acc', b_was_read = False)
        if can_s_reach_itself:
            return True
    return False


def generate_emptynesstest_dataset(automatonparameters: dict, d: int, min_acccycle_length: int = -1,
                                   max_acccycle_length: int = -1) -> list:
    """
    Takes a dictionary of automaton generation parameters, a size of the dataset and optionally a required
    range of self-reachable path lengths to further balance the dataset and returns a dataset as a
    list of torch_geometric.data.Data elements

    :param automatonparameters: a dictionary containing the parameters for the automaton generation
    :param d: the number of automata that are generated
    :param min_acccycle_length: lower bound of required length of self-reachable path of an accepting state
    :param max_acccycle_length: upper bound of required length of self-reachable path of an accepting state
    :return: a list containing a balanced dataset of d-many automata torch_geometric.data.Data elements
    """
    count = 0
    count_noacc = 0
    count_noaccfrominit = 0
    count_noaccfromacc = 0
    datalist = []
    nonempty_count = 0
    empty_count = 0
    class_by_dist_acc = []
    total_classes = (max_acccycle_length - min_acccycle_length) + 1
    for i in range(total_classes):
        class_by_dist_acc.append([])
    while (empty_count + nonempty_count) < d:
        if count % 1000 == 0:
            print(
                f"{count:d} automata have been generated so far! "
                f"Saved: {nonempty_count} nonempty - {count_noacc},{count_noaccfrominit},"
                f"{count_noaccfromacc} empty automata"
            )
        count += 1
        edge_index, edge_attr, x, acc_states = generate_nbw_er_eachsymbol(automatonparameters["nmin"],
                                                                          automatonparameters["nmax"],
                                                                          automatonparameters["pmin"],
                                                                          automatonparameters["pmax"],
                                                                          automatonparameters["paccmin"],
                                                                          automatonparameters["paccmax"],
                                                                          automatonparameters["s"],
                                                                          automatonparameters["nadd"],
                                                                          automatonparameters["featinit"])
        edge_in = [int(i) for i in edge_index[0]]
        edge_out = [int(i) for i in edge_index[1]]
        # Check non-emptyness of automaton for ground truth label
        y, type = non_emptyness_check(edge_in, edge_out, acc_states)
        if y and (nonempty_count < d / 2):
            to_acc, cycle_acc = get_minimal_distances_to_acc(edge_in, edge_out, acc_states)
            if min_acccycle_length <= cycle_acc <= max_acccycle_length:
                y = torch.tensor([0])
                data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, cyclelen=[to_acc, cycle_acc])
                if len(class_by_dist_acc[cycle_acc - min_acccycle_length]) <= d / (2 * total_classes):
                    count += 1
                    nonempty_count += 1
                    class_by_dist_acc[cycle_acc - min_acccycle_length].append(data)
            elif min_acccycle_length < 0 and max_acccycle_length < 0:
                y = torch.tensor([0])
                data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, cyclelen=[to_acc, cycle_acc])
                if len(class_by_dist_acc[0]) <= d / (2 * total_classes):
                    count += 1
                    nonempty_count += 1
                    class_by_dist_acc[0].append(data)
        elif not y and (empty_count < d / 2):
            if type == 'no_acc_state' and count_noacc <= d / 10:
                count_noacc += 1
                y = torch.tensor([1])
                empty_count += 1
                data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, cyclelen=[-1, -1])
                datalist.append(data)
            if type == 'no_acc_reached' and count_noaccfrominit <= d / 5:
                count_noaccfrominit += 1
                y = torch.tensor([1])
                empty_count += 1
                data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, cyclelen=[-2, -2])
                datalist.append(data)
            if type == 'no_acc_selfreached' and count_noaccfromacc <= d / 5:
                count_noaccfromacc += 1
                y = torch.tensor([1])
                empty_count += 1
                data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, cyclelen=[-3, -3])
                datalist.append(data)
    for c in class_by_dist_acc:
        for d in c:
            datalist.append(d)
    return datalist


def generate_bproperty_dataset_no_empty(property: str, automatonparameters: dict, d: int,
                                        minacccycle: int = -1, maxacccycle: int = -1) -> list:
    """
    aioa

    :param property: string denoting which property is recognized by the dataset:
        'min1b' checks if omega-words containing at least one b are accepted by the automata
        'infb' checks if omega-words containing infinitely-many b are accepted
    :param automatonparameters: a dictionary containing the parameters for the automaton generation
    :param d: the number of automata that are generated
    :param minacccycle: lower bound of required length of self-reachable path of an accepting state
    :param maxacccycle: upper bound of required length of self-reachable path of an accepting state
    :return: the balanced dataset as a list containing torch_geometric.data.Data elements
    """
    count = 0
    allcount = 0
    acc_count = 0
    nonacc_count = 0
    datalist = []
    class_by_dist_acc = []
    class_by_dist_nonacc = []
    total_classes = (maxacccycle - minacccycle) + 1
    for i in range(total_classes):
        class_by_dist_acc.append([])
        class_by_dist_nonacc.append([])
    while count < d:
        allcount += 1
        if allcount % 1000 == 0:
            print(f"We have generated {allcount}-many automata so far! "
                  f"acc: {acc_count}, non-acc: {nonacc_count}"
                  )
        edge_index, edge_attr, x, acc_states = generate_nbw_er_eachsymbol(automatonparameters["nmin"],
                                                                          automatonparameters["nmax"],
                                                                          automatonparameters["pmin"],
                                                                          automatonparameters["pmax"],
                                                                          automatonparameters["paccmin"],
                                                                          automatonparameters["paccmax"],
                                                                          automatonparameters["s"],
                                                                          automatonparameters["nadd"],
                                                                          automatonparameters["featinit"])
        edge_in = [int(i) for i in edge_index[0]]
        edge_out = [int(i) for i in edge_index[1]]
        # Check emptiness of automaton for ground truth label
        y, _ = non_emptyness_check(edge_in, edge_out, acc_states)
        if y:
            to_acc, cycle_acc = get_minimal_distances_to_acc(edge_in, edge_out, acc_states)
            if minacccycle <= cycle_acc <= maxacccycle:
                if property == "min1b":
                    check_accept = accepting_min1b_check(edge_in, edge_out, edge_attr, acc_states)
                else:
                    check_accept = accepting_infb_check(edge_in, edge_out, edge_attr, acc_states)
                if not check_accept:
                    y = torch.tensor([0])
                    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, cyclelen=[to_acc,cycle_acc])
                    if len(class_by_dist_nonacc[cycle_acc - minacccycle]) <= d/(2 * total_classes):
                        count += 1
                        nonacc_count += 1
                        class_by_dist_nonacc[cycle_acc - minacccycle].append(data)
                if check_accept:
                    y = torch.tensor([1])
                    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, cyclelen=[to_acc,cycle_acc])
                    if len(class_by_dist_acc[cycle_acc - minacccycle]) <= d/(2 * total_classes):
                        count += 1
                        acc_count += 1
                        class_by_dist_acc[cycle_acc - minacccycle].append(data)
    for c in class_by_dist_nonacc:
        for d in c:
            datalist.append(d)
    for c in class_by_dist_acc:
        for d in c:
            datalist.append(d)
    return datalist


def generate_bproperty_dataset_with_empty(property: str, automatonparameters: dict, d: int,
                                          minacccycle: int = -1, maxacccycle: int = -1) -> list:
    """

    :param property: string denoting which property is recognized by the dataset:
        'min1b' checks if omega-words containing at least one b are accepted by the automata
        'infb' checks if omega-words containing infinitely-many b are accepted
    :param automatonparameters: a dictionary containing the parameters for the automaton generation
    :param d: the number of automata that are generated
    :param minacccycle: lower bound of required length of self-reachable path of an accepting state
    :param maxacccycle: upper bound of required length of self-reachable path of an accepting state
    :return: the balanced dataset as a list containing torch_geometric.data.Data elements
    """
    count = 0
    allcount = 0
    acc_count = 0
    nonacc_count = 0
    empty_count = 0
    datalist = []
    class_by_dist_acc = []
    class_by_dist_nonacc = []
    total_classes = (maxacccycle - minacccycle) + 1
    for i in range(total_classes):
        class_by_dist_acc.append([])
        class_by_dist_nonacc.append([])
    while count < d:
        allcount += 1
        if allcount % 2500 == 0:
            print(f"We have generated {allcount}-many automata so far! "
                  f"acc: {acc_count}, non-acc: {nonacc_count}"
                  )
        edge_index, edge_attr, x, acc_states = generate_nbw_er_eachsymbol(automatonparameters["nmin"],
                                                                          automatonparameters["nmax"],
                                                                          automatonparameters["pmin"],
                                                                          automatonparameters["pmax"],
                                                                          automatonparameters["paccmin"],
                                                                          automatonparameters["paccmax"],
                                                                          automatonparameters["s"],
                                                                          automatonparameters["nadd"],
                                                                          automatonparameters["featinit"])
        edge_in = [int(i) for i in edge_index[0]]
        edge_out = [int(i) for i in edge_index[1]]
        # Check emptiness of automaton for ground truth label
        y, _ = non_emptyness_check(edge_in, edge_out, acc_states)
        if not y:
            # max half of the 0-labels should be empty, i.e. 1/4 of total DATASET_SIZE
            if empty_count <= d/4:
                y = torch.tensor([0])
                data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, cyclelen=[-1,-1])
                empty_count += 1
                nonacc_count += 1
                count += 1
                datalist.append(data)
        if y:
            to_acc, cycle_acc = get_minimal_distances_to_acc(edge_in, edge_out, acc_states)
            if minacccycle <= cycle_acc <= maxacccycle:
                if property == "min1b":
                    check_accept = accepting_min1b_check(edge_in, edge_out, edge_attr, acc_states)
                else:
                    check_accept = accepting_infb_check(edge_in, edge_out, edge_attr, acc_states)
                if not check_accept:
                    y = torch.tensor([0])
                    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, cyclelen=[to_acc,cycle_acc])
                    if len(class_by_dist_nonacc[cycle_acc-minacccycle]) <= d/(2*2*total_classes):
                        count += 1
                        nonacc_count += 1
                        class_by_dist_nonacc[cycle_acc-minacccycle].append(data)
                if check_accept:
                    y = torch.tensor([1])
                    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, cyclelen=[to_acc,cycle_acc])
                    if len(class_by_dist_acc[cycle_acc-minacccycle]) <= d/(2*total_classes):
                        count += 1
                        acc_count += 1
                        class_by_dist_acc[cycle_acc-minacccycle].append(data)
    for c in class_by_dist_nonacc:
        for d in c:
            datalist.append(d)
    for c in class_by_dist_acc:
        for d in c:
            datalist.append(d)
    return datalist


def generate_dataset(property: str, d: int, automatonparameters: dict,
                     minacccycle: int = 1, maxacccycle: int = 3,
                     containemptyautomata: bool = True, save_to_file: bool = True) -> list:
    """
    Generates a balanced labelled dataset of non-deterministic Büchi automata labelled whether or not they
    satisfy the given property or not: \n
    - 'empty' checks if the automaton accepts any omega-words
    - 'min1b' checks if all omega-words accepted by the automaton contain at least one b
    - 'infb' checks if all omega-words accepted by the automaton contain infinitely many b
    The automaton generation takes the given parameters and

    :param property: string denoting which property is recognized by the dataset:
        'empty' checks emptyness of the automata,
        'min1b' checks if omega-words containing at least one b are accepted by the automata,
        'infb' checks if omega-words containing infinitely-many b are accepted
    :param d: the number of automata that are generated
    :param automatonparameters: a dictionary containing the parameters for the automaton generation
    :param minacccycle: lower bound of required length of self-reachable path of an accepting state
    :param maxacccycle: upper bound of required length of self-reachable path of an accepting state
    :param containemptyautomata: if True, the dataset-elements with label 0 (i.e. not
        fulfilling the property) also contain empty automata (i.e. accepting no omega-word)
    :param save_to_file: if True, saves the generated dataset to the 'dataset_folder'
    :return: the balanced dataset as a list containing torch_geometric.data.Data elements
    """
    if property == "empty":
        datalist = generate_emptynesstest_dataset(automatonparameters, d)
    elif property == "min1b" or property == "infb":
        if containemptyautomata:
            datalist = generate_bproperty_dataset_with_empty(property, automatonparameters,
                                                            d, minacccycle=minacccycle, maxacccycle=maxacccycle)
        else:
            datalist = generate_bproperty_dataset_no_empty(property, automatonparameters,
                                                          d, minacccycle=minacccycle, maxacccycle=maxacccycle)
    else:
        print("Please give a valid property: 'empty', 'min1b', 'infb'.")
        datalist = []
        return datalist
    if save_to_file:
        torch.save(datalist, f"./{dataset_folder}/"
                             f"{property}_{d}_{str(automatonparameters['nmin'])}_{str(automatonparameters['nmax'])}")
    return datalist


paper_parameters = {"nmin": 3,
                    "nmax": 9,
                    "pmin": 0.1,
                    "pmax": 0.3,
                    "paccmin": 0.1,
                    "paccmax": 0.15,
                    "s": 2,
                    "nadd": 3,
                    "featinit": "half"
                    }

dataset = generate_dataset("infb", 20, paper_parameters)
#save_automata_from_dataset(dataset, "new", True)
