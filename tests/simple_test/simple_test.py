import numpy as np
from data_utils import *
from network_utils import *

def main():
    w2v_model, sentences = text8model(FILEPATH,SAVE_MODELS,SAVEPATH)
    simple_tsne(w2v_model, is_multi=False)


if __name__ == "__main__":
    main()
