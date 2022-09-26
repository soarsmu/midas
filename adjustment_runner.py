from metrices_calculator import *

if __name__ == '__main__':
    url_to_label, url_to_loc_mod = get_data()

    model_prob_path_java = 'probs/prob_ensemble_classifier_test_java.txt'
    model_prob_path_python = 'probs/prob_ensemble_classifier_test_python.txt'
    model_new_prob_java_path = 'probs/new_prob_java.txt'
    model_new_prob_python_path = 'probs/new_prob_python.txt'

    test_new_metric(model_prob_path_java, model_prob_path_python, model_new_prob_java_path, model_new_prob_python_path)