from metrices_calculator import *
import argparse
from variant_ensemble import infer_variant_8
import warnings
from sklearn.exceptions import DataConversionWarning
import ensemble_classifier

warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)



def do_rq1():
    url_to_label, url_to_loc_mod = get_data()

    model_new_prob_java_path = 'probs/new_prob_java.txt'
    model_new_prob_python_path = 'probs/new_prob_python.txt'

    print('-' * 64)
    print('EVALUATING JAVA DATASET')
    calculate_auc(model_new_prob_java_path, url_to_label)
    calculate_effort(model_new_prob_java_path, 'java')
    calculate_normalized_effort(model_new_prob_java_path, 'java')
    
    print('-' * 64)
    print('-' * 64)
    print('EVALUATING PYTHON DATASET')

    calculate_auc(model_new_prob_python_path, url_to_label)
    calculate_effort(model_new_prob_python_path, 'python')
    calculate_normalized_effort(model_new_prob_python_path, 'python')

def do_rq2():

    url_to_label, url_to_loc_mod = get_data()

    model_prob_path_java = 'probs/prob_ensemble_classifier_test_java.txt'
    model_prob_path_python = 'probs/prob_ensemble_classifier_test_python.txt'
    model_new_prob_java_path = 'probs/new_prob_java.txt'
    model_new_prob_python_path = 'probs/new_prob_python.txt'


    print('-' * 64)
    print('EVALUATING MiDas NO ADJUSTMENT ON JAVA DATASET')
    calculate_auc(model_prob_path_java, url_to_label)
    calculate_effort(model_prob_path_java, 'java')
    calculate_normalized_effort(model_prob_path_java, 'java')
    
    print('-' * 64)
    print('-' * 64)
    print('EVALUATING MiDas NO ADJUSTMENT ON PYTHON DATASET')

    calculate_auc(model_prob_path_python, url_to_label)
    calculate_effort(model_prob_path_python, 'python')
    calculate_normalized_effort(model_prob_path_python, 'python')

    print('-' * 64)
    print('-' * 64)
    print('-' * 64)
    print('-' * 64)

    print('-' * 64)
    print('EVALUATING MiDas ON JAVA DATASET')
    calculate_auc(model_new_prob_java_path, url_to_label)
    calculate_effort(model_new_prob_java_path, 'java')
    calculate_normalized_effort(model_new_prob_java_path, 'java')
    
    print('-' * 64)
    print('-' * 64)
    print('EVALUATING MiDas ON PYTHON DATASET')

    calculate_auc(model_new_prob_python_path, url_to_label)
    calculate_effort(model_new_prob_python_path, 'python')
    calculate_normalized_effort(model_new_prob_python_path, 'python')


def do_rq3_line():

    line_result_java = 'probs/test_ablation_line_test_java.txt'
    line_result_python = 'probs/test_ablation_line_test_python.txt'

    line_result_java_new = 'probs/test_ablation_line_test_java_new.txt'
    line_result_python_new = 'probs/test_ablation_line_test_python_new.txt'
    
    print("Extracting line-level result...")
    infer_variant_8('test_java', line_result_java, need_feature_only=False)
    infer_variant_8('test_python', line_result_python, need_feature_only=False)

    test_new_metric(line_result_java, line_result_python, line_result_java_new, line_result_python_new)

    url_to_label, url_to_loc_mod = get_data()

    print("Evaluating MiDas line level...")

    print('-' * 64)
    print('EVALUATING JAVA DATASET')
    calculate_auc(line_result_java_new, url_to_label)
    calculate_effort(line_result_java_new, 'java')
    calculate_normalized_effort(line_result_java_new, 'java')
    
    print('-' * 64)
    print('-' * 64)
    print('EVALUATING PYTHON DATASET')

    calculate_auc(line_result_python_new, url_to_label)
    calculate_effort(line_result_python_new, 'python')
    calculate_normalized_effort(line_result_python_new, 'python')


def do_rq3_line_hunk():
    url_to_label, url_to_loc_mod = get_data()

    model_prob_path_java = 'probs/test_ablation_line_hunk_java.txt'
    model_prob_path_python = 'probs/test_ablation_line_hunk_python.txt'
    model_new_prob_java_path = 'probs/test_ablation_line_hunk_java_new.txt'
    model_new_prob_python_path = 'probs/test_ablation_line_hunk_python_new.txt'

    test_new_metric(model_prob_path_java, model_prob_path_python, model_new_prob_java_path, model_new_prob_python_path)

    print('-' * 64)
    print('EVALUATING MiDas line_level + hunk_level ON JAVA DATASET')
    calculate_auc(model_new_prob_java_path, url_to_label)
    calculate_effort(model_new_prob_java_path, 'java')
    calculate_normalized_effort(model_new_prob_java_path, 'java')
    
    print('-' * 64)
    print('-' * 64)
    print('EVALUATING MiDas line_level + hunk_level ON PYTHON DATASET')

    calculate_auc(model_new_prob_python_path, url_to_label)
    calculate_effort(model_new_prob_python_path, 'python')
    calculate_normalized_effort(model_new_prob_python_path, 'python')


def do_rq3_line_hunk_file():
    url_to_label, url_to_loc_mod = get_data()

    model_prob_path_java = 'probs/test_ablation_line_hunk_file_java.txt'
    model_prob_path_python = 'probs/test_ablation_line_hunk_file_python.txt'
    model_new_prob_java_path = 'probs/test_ablation_line_hunk_file_java_new.txt'
    model_new_prob_python_path = 'probs/test_ablation_line_hunk_file_python_new.txt'

    test_new_metric(model_prob_path_java, model_prob_path_python, model_new_prob_java_path, model_new_prob_python_path)

    print('-' * 64)
    print('EVALUATING MiDas line_level + hunk_level + file_level ON JAVA DATASET')
    calculate_auc(model_new_prob_java_path, url_to_label)
    calculate_effort(model_new_prob_java_path, 'java')
    calculate_normalized_effort(model_new_prob_java_path, 'java')
    
    print('-' * 64)
    print('-' * 64)
    print('EVALUATING MiDas line_level + hunk_level + file_level ON PYTHON DATASET')

    calculate_auc(model_new_prob_python_path, url_to_label)
    calculate_effort(model_new_prob_python_path, 'python')
    calculate_normalized_effort(model_new_prob_python_path, 'python')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Ensemble Classifier')
    parser.add_argument('--rq',
                        type=int,
                        default=1,
                        help='research question number, from 1')
    parser.add_argument('--mode',
                        type=int,
                        default=1,
                        help='mode for RQ (rq3)',
                        required=False)

    args = parser.parse_args()
    rq = args.rq
    mode = args.mode
    if rq == 1:
        do_rq1()
    elif rq == 2:
        do_rq2()
    if rq == 3:
        if mode == 1:
            do_rq3_line()
        elif mode == 2:
            do_rq3_line_hunk()
        elif mode == 3:
            do_rq3_line_hunk_file()
    else:
        raise Exception("Invalid RQ number")

    