These are the files containing the Convex Hull and Traveling Salesman Problem dataset present in the “Pointer Networks” paper:

Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. "Pointer networks." NIPS 2015

@inproceedings{vinyals2015pointer,
  title={Pointer networks},
  author={Vinyals, Oriol and Fortunato, Meire and Jaitly, Navdeep},
  booktitle={NIPS},
  pages={2692--2700},
  year={2015}
}

The following is an example from one of the lines from one of the files. It contains the 2d coordinates (in the format x_1 y_1 x_2 y_2 … x_N y_N) followed by the word “output” as a separator, and then the indices of the corresponding outputs (base 1) of the input points. In the example below, the convex hull was 1 2 3 4 5 1 (note that the last digit is repeated as it forms a tour), i.e., first point (0.99624295 0.92802603), followed by second point (0.24557767 0.80411435), etc.

0.99624295 0.92802603 0.24557767 0.80411435 0.43667577 0.33301639 0.36675234 0.71464094 0.80695481 0.14914953 output 1 2 3 5 1 

The name of the files should be self explanatory (some contain a range of inputs points N).

Feel free to contact us if you have any questions!

