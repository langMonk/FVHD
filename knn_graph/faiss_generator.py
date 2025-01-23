import struct
import sys
from datetime import datetime

import faiss
import numpy as np
import pandas as pd
from numpy import sqrt


class FaissGenerator:
    def __init__(
        self,
        df: pd.DataFrame,
        cosine_metric: bool = False,
        binary_metric: bool = False,
        device="auto",
    ):
        self.indexes = []
        self.distances = []
        self.nn = None
        self.cosine_metric = (cosine_metric,)
        self.binary_metric = (binary_metric,)
        if device in ["cuda", "auto"] and (num_gpus := faiss.get_num_gpus()) >= 1:
            print(f"Faiss found {num_gpus} GPU(s). We only need one :D")
            # For emnist gpu is 10x faster (4s compared to 40s)
            self.__gpu_res = faiss.StandardGpuResources()
        else:
            self.__gpu_res = None
        self.N = len(df.axes[0])
        self.M = len(df.axes[1]) - 1

        if self.cosine_metric:
            # if cosine - we have to normalize the dataset
            self.X = df.to_numpy(dtype="float").astype("float32")[:, 1:]
            norm = np.linalg.norm(self.X, axis=1).reshape(-1, 1)
            self.X = np.divide(self.X, norm)
        else:
            self.X = df.to_numpy(dtype="float").astype("float32")[:, 1:]

        self.X = self.X.copy(order="C")

    def run(self, nn: int = 100):
        self.nn = nn

        index_flat = None
        if self.cosine_metric:
            quantizer = faiss.IndexFlatIP(self.M)
            index_flat = faiss.IndexIVFFlat(quantizer, self.M, int(sqrt(self.N)))
            index_flat.nprobe = 10
        else:
            quantizer = faiss.IndexFlatL2(self.M)
            index_flat = faiss.IndexIVFFlat(quantizer, self.M, int(sqrt(self.N)))

        if self.__gpu_res:
            index_flat = faiss.index_cpu_to_gpu(self.__gpu_res, 0, index_flat)
        assert not index_flat.is_trained
        index_flat.train(self.X)
        assert index_flat.is_trained

        index_flat.add(self.X)

        start = datetime.now()
        print("Searching...")
        index_flat.nprobe = 10
        distances, self.indexes = index_flat.search(self.X, nn + 1)
        print("Finished.")

        print(datetime.now() - start, file=sys.stderr)

        # normalize distances
        norm = np.linalg.norm(self.X)
        self.distances = np.divide(distances, norm)

        if self.binary_metric:
            max_indices = np.argmax(distances, axis=1)
            self.distances = np.zeros_like(distances)

            self.distances[np.arange(distances.shape[0]), max_indices] = 1

        return self.distances, self.indexes

    def save_to_binary_file(self, output_file_path):
        with open(output_file_path, "wb") as f:
            f.write("{};{};{}\n".format(self.N, self.nn, 8).encode("ascii"))
            f.write(0x01020304.to_bytes(8, byteorder="little"))
            for i in range(0, len(self.indexes)):
                for j in range(0, len(self.indexes[i])):
                    if i != self.indexes[i][j]:
                        f.write(int(self.indexes[i][j]).to_bytes(8, byteorder="little"))
                        f.write(bytearray(struct.pack("f", self.distances[i][j])))
            f.close()

    def save_to_text_file(self, output_file_path):
        with open(output_file_path, "w") as f:
            f.write(str(self.N) + "\n")
            f.write(str(self.nn) + "\n")
            f.write("0x01020304" + "\n")
            for i in range(0, len(self.indexes)):
                for j in range(0, len(self.indexes[i])):
                    if i != self.indexes[i][j]:
                        f.write(
                            "{} {} {}\n".format(
                                i, self.indexes[i][j], self.distances[i][j]
                            )
                        )


if __name__ == "__main__":
    dataset_path = "./datasets/emnist_letters_data_pca_100.csv"
    generator = FaissGenerator(dataset_path, cosine_metric=True)
    dist, ind = generator.run(nn=100)
    generator.save_to_binary_file("emnist_letters_data_pca_100_cosine.bin")
    # generator.save_to_text_file("mnist_pca_100_euclidean.txt")
