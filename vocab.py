import torch

class Vocab(object):
    def __init__(self, data=None, lower=False):
        self.idx_to_sym = {}
        self.sym_to_idx = {}
        self.frequencies = {}
        self.lower = lower

        # Special entries will not be pruned.
        self.special = []

        if data is not None:
            if type(data) == str:
                self.load_file(data)
            else:
                self.add_specials(data)

    def size(self):
        return len(self.idx_to_sym)

    # Load entries from a file.
    def load_file(self, filename):
        for line in open(filename):
            fields = line.split()
            sym = fields[0]
            idx = int(fields[1])
            self.add(sym, idx)

    # Write entries to a file.
    def write_file(self, filename):
        with open(filename, 'w') as file:
            for i in range(self.size()):
                sym = self.idx_to_sym[i]
                file.write('%s %d\n' % (sym, i))

        file.close()

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.sym_to_idx[key]
        except KeyError:
            return default

    def get_sym(self, idx, default=None):
        try:
            return self.idx_to_sym[idx]
        except KeyError:
            return default

    # Mark this `sym` and `idx` as special (i.e. will not be pruned).
    def add_special(self, sym, idx=None):
        idx = self.add(sym, idx)
        self.special += [idx]

    # Mark all syms in `syms` as specials (i.e. will not be pruned).
    def add_specials(self, syms):
        for sym in syms:
            self.add_special(sym)

    # Add `sym` in the dictionary. Use `idx` as its index if given.
    def add(self, sym, idx=None):
        if idx is not None:
            self.idx_to_sym[idx] = sym
            self.sym_to_idx[sym] = idx
        else:
            if sym in self.sym_to_idx:
                idx = self.sym_to_idx[sym]
            else:
                idx = len(self.idx_to_sym)
                self.idx_to_sym[idx] = sym
                self.sym_to_idx[sym] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    
    # Return a new dictionary with vocabs >= min_freq.
    def prune_by_freq(self, min_freq):
        # Sort by frequency
        freq = torch.Tensor(
                [self.frequencies[i] for i in range(len(self.frequencies))])
        sort_freq, idx = torch.sort(freq, 0, True)

        new_dict = Vocab()

        # Add special entries in all cases.
        for i in self.special:
            new_dict.add_special(self.idx_to_sym[i])

        for f, i in zip(sort_freq, idx):
            if f < min_freq:
                break
            new_dict.add(self.idx_to_sym[i])

        return new_dict

        if size >= self.size():
            return self

        # Only keep the `size` most frequent entries.
        freq = torch.Tensor(
                [self.frequencies[i] for i in range(len(self.frequencies))])
        _, idx = torch.sort(freq, 0, True)

        new_dict = Vocab(lower=self.lower)

        # Add special entries in all cases.
        for i in self.special:
            new_dict.add_special(self.idx_to_sym[i])

        for i in idx[:size]:
            new_dict.add(self.idx_to_sym[i])

        return new_dict

    # Convert `symbols` to indices (LongTensor). Use `unk` if not found.
    # Optionally insert `bos` at the beginning and `eos` at the .
    def convert_to_idx(self, syms, unk, bos=None, eos=None):
        vec = []

        if bos is not None:
            vec += [self.lookup(bos)]

        unk = self.lookup(unk)
        # vec += [self.lookup(sym, default=unk) for sym in syms]
        for sym in syms:
            idx = self.lookup(sym, default=unk)
            vec.append(idx)

        if eos is not None:
            vec += [self.lookup(eos)]

        return torch.LongTensor(vec)

    # Convert list of indices to list of syms. If index `stop` is reached, convert it and return.
    def convert_to_sym(self, idx, stop=None):
        syms = []

        for i in idx:
            syms.append(self.get_sym(i))
            if stop is not None and i == stop:
                break

        return syms
