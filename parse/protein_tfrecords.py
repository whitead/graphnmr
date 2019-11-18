import tensorflow as tf
import pickle
import numpy as np
import os
from pdbfixer import PDBFixer
from simtk.openmm import app
import Bio.SeqUtils as seq
from Bio import pairwise2
from simtk import unit
import gsd.hoomd
import math
import tqdm, os
import random
import traceback
from model import *

def process_corr(path, debug):
    with open(path, 'r') as f:
        peaks = []
        sequence_map = {}
        sequence = []
        entry_lines = False
        sequence_lines = False
        mapping_mode = None
        index = 0
        last_id = -1
        for line in f.readlines():
            if '_Chem_shift_ambiguity_code' in line:
                entry_lines = True
                continue
            elif mapping_mode is None and '_Residue_label' in line:
                sequence_lines = True
                continue

            if entry_lines and 'stop_' in line:
                entry_lines = False
                continue
            elif sequence_lines and 'stop_' in line:
                sequence_lines = False
                continue
            if entry_lines and len(line.split()) > 0:
                _,srid,rname,name,element,shift,_,_ = line.split()
                rid = int(srid)
                if rid != last_id:
                    peaks.append(dict(name=rname))
                    last_id = rid
                peaks[-1][name] = shift
                peaks[-1]['index'] = srid
            elif sequence_lines and len(line.split()) > 0:
                sline = line.split()
                if mapping_mode is None:
                    try:
                        int(sline[1])
                        mapping_mode = True
                    except:
                        mapping_mode = False

                if mapping_mode:
                    for i in range(len(sline) // 3):
                        pid = int(sline[i * 3]) - 1
                        while len(sequence) <= pid:
                            sequence.append('XXX')
                        sequence[pid] = sline[i * 3 + 2]

                else:
                    for i in range(len(sline) // 2):
                        index = int(sline[i * 2])
                        while len(sequence) < index:
                            sequence.append('XXX')
                        sequence[index - 1] = sline[i * 2 + 1]

    if len(peaks) == 0:
        raise ValueError('Could not parse file')

    for i,p in enumerate(peaks):
        sequence_map[int(p['index']) - 1] = i

    return peaks,sequence_map,sequence

def align(seq1, seq2):
    flat1 = seq.seq1(''.join(seq1)).replace('X', '-')
    flat2 = seq.seq1(''.join(seq2)).replace('X', '-')
    flats = [flat1, flat2]
    # aligning 2 to 1 seems to give better results
    align = pairwise2.align.localxx(flat2, flat1, one_alignment_only=True)
    start = align[0][3]
    offset = [0,0]
    # compute how many gaps had to be inserted at beginning to align
    for i in range(2):
        for j in range(len(align[0][0])):
            # account for the fact that 2 and 1 are switched in alignment results
            if align[0][(i + 1) % 2][j] == '-':
                if flats[i][j - offset[i]] != '-':
                    offset[i] += 1
            else:
                break
    return -offset[0], -offset[1]

# NN is not NEIGHBOR_NUMBer
# reason for difference is we don't want 1,3 or 1,4, etc neighbors on the list
def process_pdb(path, corr_path, chain_id, max_atoms,
                gsd_file, embedding_dicts, NN, nlist_model,  
                keep_residues=[-1, 1],
                debug=False, units = unit.nanometer, frame_number=3):
    # load pdb
    pdb = app.PDBFile(path)

    # load cs sets
    peak_data, sequence_map, peak_seq = process_corr(corr_path, debug)
    result = []
    # check for weird/null chain
    if chain_id == '_':
        chain_id = 'A'
    residues = list(filter(lambda r: r.chain.id == chain_id, pdb.topology.residues()))
    pdb_offset, seq_offset = None, None

    # from pdb residue index to our aligned residue index
    residue_lookup = {}
    # bonded neighbor mask
    nlist_mask = None
    peak_count = 0
    # select a random set of frames for generating data.
    frame_choices = random.choices(range(0, pdb.getNumFrames()), k=frame_number)
    for fi in frame_choices:
        successes = 0
        # clean up individual frame 
        frame = pdb.getPositions(frame=fi)
        # have to fix at each frame since inserted atoms may change
        # fix missing residues/atoms
        fixer = PDBFixer(filename=path)
        # overwrite positions with frame positions
        fixer.positions = frame
        # we want to add missing atoms,
        # but not replace missing residue. We'd 
        # rather just ignore those
        fixer.findMissingResidues()
        # remove the missing residues
        fixer.missingResidues = []
        # remove water!
        fixer.removeHeterogens(False)
        fixer.findMissingAtoms()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        # get new positions
        frame = fixer.positions
        num_atoms = len(frame)
        if num_atoms > 10000:
            if debug:
                print('Exceeded number of atoms for building nlist (change this if you have big GPU memory')
            break
        # remake residue list each time so they have correct atom ids
        residues = list(filter(lambda r: r.chain.id == chain_id, fixer.topology.residues()))
        # check alignment once
        if pdb_offset is None:
            # create sequence from residues
            pdb_seq = ['XXX'] * max([int(r.id) + 1 for r in residues])
            for r in residues:
                pdb_seq[int(r.id)] = r.name
            pdb_offset, seq_offset = align(pdb_seq, peak_seq)

            # create resiud look-up from atom index
            for i,r in enumerate(residues):
                for a in r.atoms():
                    residue_lookup[a.index] = i
            # This alignment will be checked as we compare shifts against the pdb
        # get neighbor list for frame
        np_pos = np.array([v.value_in_unit(units) for v in frame])
        frame_nlist = nlist_model(np_pos)

        for ri in range(len(residues)):
            # we build up fragment by getting residues around us, both in chain 
            # and those within a certain distance of us
            rmin = max(0,ri + keep_residues[0])
            rmax = min(len(residues), ri + keep_residues[1] + 1)
            # do we have any residues to consider?
            success = rmax - rmin > 0

            consider = set(range(rmin, rmax))
            # now grab spatial neighbor residues
            for a in residues[ri].atoms():
                for ni in range(NN):
                    j = int(frame_nlist[a.index, ni, 1])
                    try:
                        consider.add(residue_lookup[j])
                    except KeyError as e:
                        success = False
                        if debug:
                            print('Neighboring residue in different chain, skipping')
                        break
            atoms = np.zeros((max_atoms), dtype=np.int64)
            # we will put dummy atom at end that fills-up special neighbors
            atoms[-1] = embedding_dicts['atom']['Z']
            mask = np.zeros( (max_atoms), dtype=np.float)
            bonds = np.zeros( (BOND_MAX, max_atoms, max_atoms), dtype=np.int64)
            # nlist:
            # :,:,0 -> distance
            # :,:,1 -> neighbor index
            # :,:,2 -> bond count
            nlist = np.zeros( (max_atoms, NEIGHBOR_NUMBER, 3), dtype=np.float)
            positions = np.zeros( (max_atoms, 3), dtype=np.float)
            peaks = np.zeros( (max_atoms), dtype=np.float)
            names = np.zeros( (max_atoms), dtype=np.int64)
            # going from pdb atom index to index in these data structures
            rmap = dict()
            index = 0
            # check our two conditions that could have made this false: there are residues and 
            # we didn't have off-chain spatial neighboring residues
            if not success:
                continue
            for rj in consider:
                residue = residues[rj]
                # use the alignment result to get offset
                segid = int(residue.id) + pdb_offset
                if segid + seq_offset not in sequence_map:
                    if debug:
                        print('Could not find residue index', rj, ': ', residue, 'in the sequence map. Its index is', segid + seq_offset, 'ri: ', ri)
                        print(sequence_map)
                    success = False
                    break
                peak_id = sequence_map[segid + seq_offset]
                #peak_id = segid
                if peak_id >= len(peak_data):
                    success = False
                    break
                # only check for residue we actually care about
                if ri == rj and residue.name != peak_data[peak_id]['name']:
                    if debug:
                        print('Mismatch between residue ', ri, rj, peak_id, residue, segid, peak_data[peak_id], path, corr_path, chain_id)
                    success = False
                    break
                for atom in residue.atoms():
                    mask[index] = float(ri == rj)
                    atom_name = residue.name + '-' + atom.name
                    if atom_name not in embedding_dicts['name']:
                        embedding_dicts['name'][atom_name] = len(embedding_dicts['name'])
                    names[index] = embedding_dicts['name'][atom_name]

                    if atom.element.symbol not in embedding_dicts['atom']:
                        if debug:
                            print('Could not identify atom', atom.element.symbol)
                        success = False
                        break
                    atoms[index] = embedding_dicts['atom'][atom.element.symbol]
                    positions[index] = np_pos[atom.index, :]
                    rmap[atom.index] = index
                    peaks[index] = 0
                    if mask[index]:
                        if atom.name[:3] in peak_data[peak_id]:
                            peaks[index] = peak_data[peak_id][atom.name[:3]]
                            peak_count += 1
                        else:
                            mask[index] = 0
                    index += 1
                    # -1 for dummy atom which is stored at end
                    if index == max_atoms - 2:
                        if debug:
                            print('Not enough space for all atoms')
                        success = False
                        break
                if ri == rj and sum(mask) == 0:
                    if debug:
                        print('Warning found no peaks for', ri, rj, residue, peak_data[peak_id])
                    success = False
                if not success:
                    break
            if not success:
                continue
            # do this after so our reverse mapping is complete
            for rj in consider:
                residue = residues[rj]
                for b in residue.bonds():
                    # use distance as bond
                    try:
                        bonds[0,rmap[b.atom1.index], rmap[b.atom2.index]] = 1
                        bonds[0,rmap[b.atom2.index], rmap[b.atom1.index]] = 1
                    except KeyError:
                        # for bonds that cross residue
                        pass
            # bonds contains 1 neighbors, 2 neighbors, etc where "1" means 1 bond away and "2" means two bonds away
            for bi in range(1, BOND_MAX):
                bonds[bi, :, :] = (np.matmul(bonds[0, :, :], bonds[bi - 1, :, :])) > 0

            for rj in consider:
                residue = residues[rj]
                for a in residue.atoms():
                    index = rmap[a.index]
                    # convert to local indices and filter neighbors
                    n_index = 0
                    for ni in range(NN):
                        if frame_nlist[a.index, ni,0] > 50.0:
                            # large distances are sentinels for things
                            # like self neighbors
                            continue
                        try:
                            j = rmap[int(frame_nlist[a.index, ni, 1])]
                        except KeyError:
                            # either we couldn't find a neighbor on the root residue (which is bad)
                            # or just one of the neighbors is not on a considered residue. 
                            if rj == ri:
                                success = False
                                if debug:
                                    print('Could not find all neighbors', int(frame_nlist[a.index, ni, 1]), consider)
                                break
                            j = max_atoms - 1 # point to dummy atom
                        # mark as not a neighbor if out of molecule (only for non-subject nlists)
                        if j == max_atoms - 1:
                            #set index
                            nlist[index,n_index,1] = j
                            # set distance
                            nlist[index,n_index,0] = frame_nlist[a.index, ni,0]
                            #set type
                            nlist[index, n_index, 2] = embedding_dicts['nlist']['none']
                            n_index += 1
                        # a 0 -> non-bonded
                        elif sum(bonds[:, index, j]) == 0:
                            #set index
                            nlist[index,n_index,1] = j
                            # set distance
                            nlist[index,n_index,0] = frame_nlist[a.index, ni,0]
                            #set type
                            nlist[index,n_index,2] = embedding_dicts['nlist']['nonbonded']
                            n_index += 1
                        # value of 0 -> single bonded
                        elif (bonds[:, index, j] != 0).argmax(0) == 0:
                            #set index
                            nlist[index,n_index,1] = j
                            # set distance
                            nlist[index,n_index,0] = frame_nlist[a.index,ni,0]
                            #set type
                            nlist[index,n_index,2] = embedding_dicts['nlist'][1]
                            n_index += 1
                        if n_index == NEIGHBOR_NUMBER:
                            break
                    # how did we do on peaks
                    if False and (peaks[index] > 0 and peaks[index] < 25):
                        nonbonded_count =  np.sum(nlist[index, :, 2] == embedding_dicts['nlist']['nonbonded'])
                        bonded_count = np.sum(nlist[index, :, 2] == embedding_dicts['nlist'][1])
                        print('neighbor summary: non-bonded: {}, bonded: {}, total: {}'.format(nonbonded_count, bonded_count, NEIGHBOR_NUMBER))
                        print(nlist[index, :, :])
                        exit()
            if not success:
                continue
            if gsd_file is not None:
                snapshot = write_record_traj(positions, atoms, mask, nlist, peaks, embedding_dicts['class'][residues[ri].name], names, embedding_dicts)
                snapshot.configuration.step = successes
                gsd_file.append(snapshot)
            result.append(make_tfrecord(atoms, mask, nlist, peaks, embedding_dicts['class'][residues[ri].name], names))
            successes += 1
    return result, successes / len(peak_data), len(result), peak_count


PROTEIN_DIR = 'data/proteins/'
WRITE_FRAG_PERIOD = 25

# load embedding information
embedding_dicts = load_embeddings('embeddings.pb')

# load data info
with open(PROTEIN_DIR + 'data.pb', 'rb') as f:
    protein_data = pickle.load(f)


items = list(protein_data.values())
results = []
records = 0
peaks = 0
# turn off GPU for more memory
config = tf.ConfigProto(
       # device_count = {'GPU': 0}
    )
#config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
with tf.python_io.TFRecordWriter('train-structure-protein-data-{}-{}.tfrecord'.format(MAX_ATOM_NUMBER, NEIGHBOR_NUMBER),
                                 options=tf.io.TFRecordCompressionType.GZIP) as writer:
    with tf.Session(config=config) as sess,\
        gsd.hoomd.open(name='protien_frags.gsd', mode='wb') as gsd_file:
        # multiply x 8 in case we have some 1,3, or 1,4 neighbors that we don't want
        NN = NEIGHBOR_NUMBER * 8
        nm = nlist_model(NN, sess)
        pbar = tqdm.tqdm(items)

        for index, entry in enumerate(pbar):
            try:
                result, p, n, pc = process_pdb(PROTEIN_DIR + entry['pdb_file'], PROTEIN_DIR + entry['corr'], entry['chain'], 
                                        gsd_file=gsd_file if index % WRITE_FRAG_PERIOD == 0 else None,
                                        max_atoms=MAX_ATOM_NUMBER, embedding_dicts=embedding_dicts, NN=NN, 
                                        nlist_model=nm)
                pbar.set_description('Processed PDB {} ({}). Successes {} ({:.2}). Total Records: {}, Peaks: {}. Wrote frags: {}'.format(
                                   entry['pdb_id'], entry['corr'], n, p, records, peaks, index % WRITE_FRAG_PERIOD == 0))
                for r in result:
                    writer.write(r.SerializeToString())
                records += n
                peaks += pc
                save_embeddings(embedding_dicts, 'embeddings.pb')
            except (ValueError, IndexError) as e:
                print(traceback.format_exc())
                pbar.set_description('Failed in ' +  entry['pdb_id'], entry['corr'])
print('wrote ', records)