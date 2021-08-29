from model.DHIM import DHIM

if __name__ == "__main__":
    argparser = DHIM.get_model_specific_argparser()
    hparams = argparser.parse_args()
    model = DHIM(hparams)
    if hparams.train:
        model.run_training_sessions()
    else:
        model.load()
        print('Loaded model with: %s' % model.flag_hparams())

        # model.run_retrieval_case_study()  # retrieval case study
        # model.hash_code_visualization()   # hash code visualization

        val_perf, test_perf = model.run_test()
        print('Val:  {:8.2f}'.format(val_perf))
        print('Test: {:8.2f}'.format(test_perf))