from typing import List

import numpy as np


class OneHotEncoder:

    def __init__(self, padder: str, max_length: int = None):
        """
        Parameters
        ----------
        padder: str
            Character to perform padding with.
        max_length: int
            Maximum length of the sequences.
        """
        self.padder = padder
        self.max_length = max_length

        self.alphabet = None  #The unique characters in the sequences
        self.char_to_index = None  #Dictionary mapping characters in the alphabet to unique integers
        self.index_to_char = None  #Reverse of char_to_index (dictionary mapping integers to characters)


    def fit(self, data: List) -> None:
        """
        Fits the encoder to the data.

        Parameters
        ----------
        data: List
            List of sequences (strings) to learn from.

        """

        self.alphabet = sorted(set(''.join(data)))
        self.char_to_index = {char: index for index, char in enumerate(self.alphabet)}
        self.index_to_char = {index: char for index, char in enumerate(self.alphabet)}

        if self.max_length is None:
            self.max_length = max(len(seq) for seq in data)

    def transform(self, data: List) -> List[np.ndarray]:
        """
        Encodes the sequence to one hot encoding.

        Parameters
        ----------
        data: List
            Data to encode.

        Returns
        -------
        List
            Encoded sequences.
        """

        enconded_seqs = []

        for seq in data:

            if len(seq) > self.max_length:        #Trim the sequences to max_length
                seq = seq[:self.max_length]

            elif len(seq) < self.max_length:      #Pad the sequences with the padding character
                seq_padding = self.padder * (self.max_length - len(seq))
                seq = seq + seq_padding

            #Encode the data to the one hot encoded matrices
            enc_seq = np.zeros((self.max_length, len(self.alphabet)))   #Matrix of shape max_length x alphabet_size

            for x, char in enumerate(seq):
                if char in self.char_to_index:
                    enc_seq[x, self.char_to_index[char]] = 1
            enconded_seqs.append(enc_seq)
        
        return enconded_seqs

    def fit_transform(self, data: List) -> List[np.ndarray]:
        """
        Runs fit and then predict.

        Parameters
        ----------
        data: List
            List of sequences (strings) to learn from.
        
        Returns
        -------
        List
            
        """

        self.fit(data)
        return self.transform(data)
        

    def inverse_transform(self, encoded_sequences: List) -> List[str]:
        """
        Converts one-hot-encoded sequences back to sequences.

        Parameters
        ----------
        encoded_sequences: List
            Data to decode (one-hot encoded matrices).

        """

        decoded_seqs = []

        for enc_seq in encoded_sequences:
            dec_seq = ''.join([self.index_to_char[index] for index in enc_seq.argmax(axis=1)])
            decoded_seqs.append(dec_seq)
        
        return decoded_seqs

