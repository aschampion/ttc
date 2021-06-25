import labeling
import sys

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    pred = labeling.Predictor(labeling.FormatNetConfig('nuclei'), '/volumes/labels/nuclei/s0')
    pred.predict(3000000, int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
