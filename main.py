import argparse
from botPlayer2 import play
from botTrainer2 import train
from dataCollector import DataCollector
from botReTrainer import re_train




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a tool for collecting training data, \
    training a neural network model collected data, and running a bot using the output of the neural network.")
    parser.add_argument("-e", "--eta", help="Trains the model with a learning rate set to the value of this argument", type=float)
    parser.add_argument("-d", "--dataset", help="Trains the model on the passed relative path to the dataset.", type=str)
    parser.add_argument("-m", "--model", help="Saves or loads the model in the path under this argument.", type=str)
    parser.add_argument("-l", "--loadpath", help="Saves or loads the model in the path under this argument.", type=str)
    parser.add_argument("-b", "--batchsize", help="Trains the model with the given batch size.", type=int)
    parser.add_argument("-x", "--chunksize", help="Loads data in chunks the size of this arg.", type=int)
    parser.add_argument("-c", "--collect", help="Puts the program in data collection mode. (OPT: --dataset dataset_name)", action="store_true")
    parser.add_argument("-t", "--train", help="Puts the program in training mode. (REQ: --dataset dataset_name. OPT: --model model_name)", action="store_true")
    parser.add_argument("-r", "--retrain", help="Puts the program in training mode. (REQ: --loadpath load_path, --dataset dataset_name --model model_name)", action="store_true")
    parser.add_argument("-p", "--play", help="Puts the program in playing mode. (REQ: --model model_name)", action="store_true")
    args = parser.parse_args()
    def validate(illegal, required):
        assert(all(not x for x in illegal))
        assert(all(x for x in required))
    if args.collect:
        print("Collecting Data Selected")
        illegal = [args.train, args.play, args.model, args.eta]
        required = []
        validate(illegal, required)
        dataCollector = DataCollector(args.dataset)
        dataCollector.run()
    elif args.train:
        print("Train Model Selected")
        illegal = [args.play, args.collect]
        required = [args.dataset]
        validate(illegal, required)
        train(args.dataset, model_save_name=args.model, chunk_size=None)
    elif args.retrain:
        print("ReTrain Model Selected")
        illegal = [args.play, args.collect]
        required = [args.loadpath, args.dataset]
        validate(illegal, required)
        re_train(args.dataset, args.loadpath, model_save_name=args.model, chunk_size=None)
    elif args.play:
        print("Play Model Selected")
        illegal = [args.collect, args.train, args.eta]
        required = [args.model]
        validate(illegal, required)
        play(args.model)
    else:
        raise Exception("Error: No valid argument for program operation.\
            \nUse --collect, --train, or --play to start with an \
            operation.")
