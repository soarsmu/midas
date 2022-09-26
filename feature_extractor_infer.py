from variant_ensemble import *

if __name__ == '__main__':
    print("Inferring variant 1...")
    infer_variant_1('train', 'features/feature_variant_1_train.txt', need_feature_only=True)
    infer_variant_1('test_java', 'features/feature_variant_1_test_java.txt', need_feature_only=True)
    infer_variant_1('test_python', 'features/feature_variant_1_test_python.txt', need_feature_only=True)
    print('-' * 64)
    
    print("Inferring variant 2...")
    infer_variant_2('train', 'features/feature_variant_2_train.txt', need_feature_only=True)
    infer_variant_2('test_java', 'features/feature_variant_2_test_java.txt', need_feature_only=True)
    infer_variant_2('test_python', 'features/feature_variant_2_test_python.txt', need_feature_only=True)
    print('-' * 64)

    print("Inferring variant 3...")
    infer_variant_3('train', 'features/feature_variant_3_train.txt', need_feature_only=True)
    infer_variant_3('test_java', 'features/feature_variant_3_test_java.txt', need_feature_only=True)
    infer_variant_3('test_python', 'features/feature_variant_3_test_python.txt', need_feature_only=True)
    
    print("Inferring variant 5...")
    infer_variant_5('train', 'features/feature_variant_5_train.txt', need_feature_only=True)
    infer_variant_5('test_java', 'features/feature_variant_5_test_java.txt', need_feature_only=True)
    infer_variant_5('test_python', 'features/feature_variant_5_test_python.txt', need_feature_only=True)
    print('-' * 64)
    
    print("Inferring variant 6...")
    infer_variant_6('train', 'features/feature_variant_6_train.txt', need_feature_only=True)
    infer_variant_6('test_java', 'features/feature_variant_6_test_java.txt', need_feature_only=True)
    infer_variant_6('test_python', 'features/feature_variant_6_test_python.txt', need_feature_only=True)
    
    print("Inferring variant 7...")
    infer_variant_7('train', 'features/feature_variant_7_train.txt', need_feature_only=True)
    infer_variant_7('test_java', 'features/feature_variant_7_test_java.txt', need_feature_only=True)
    infer_variant_7('test_python', 'features/feature_variant_7_test_python.txt', need_feature_only=True)

    print("Inferring variant 8...")
    infer_variant_8('train', 'features/feature_variant_8_train.txt', need_feature_only=True)
    infer_variant_8('test_java', 'features/feature_variant_8_test_java.txt', need_feature_only=True)
    infer_variant_8('test_python', 'features/feature_variant_8_test_python.txt', need_feature_only=True)


    # print("Inferring variant 2 CNN...")
    # infer_variant_2_cnn('train', 'features/feature_variant_2_cnn_train.txt', need_feature_only=True)
    # infer_variant_2_cnn('test_java', 'features/feature_variant_2_cnn_test_java.txt', need_feature_only=True)
    # infer_variant_2_cnn('test_python', 'features/feature_variant_2_cnn_test_python.txt', need_feature_only=True)


    # print("Inferring variant 6 CNN...")
    # infer_variant_6_cnn('train', 'features/feature_variant_6_cnn_train.txt', need_feature_only=True)
    # infer_variant_6_cnn('test_java', 'features/feature_variant_6_cnn_test_java.txt', need_feature_only=True)
    # infer_variant_6_cnn('test_python', 'features/feature_variant_6_cnn_test_python.txt', need_feature_only=True)

    # print("Inferring variant 8 LSTM...")
    # infer_variant_8_lstm('train', 'features/feature_variant_8_lstm_train.txt', need_feature_only=True)
    # infer_variant_8_lstm('test_java', 'features/feature_variant_8_lstm_test_java.txt', need_feature_only=True)
    # infer_variant_8_lstm('test_python', 'features/feature_variant_8_lstm_test_python.txt', need_feature_only=True)

    # print("Inferring variant 8 GRU...")
    # infer_variant_8_gru('train', 'features/feature_variant_8_gru_train.txt', need_feature_only=True)
    # infer_variant_8_gru('test_java', 'features/feature_variant_8_gru_test_java.txt', need_feature_only=True)
    # infer_variant_8_gru('test_python', 'features/feature_variant_8_gru_test_python.txt', need_feature_only=True)


    # print("Inferring variant 3 FCN...")
    # infer_variant_3_fcn('train', 'features/feature_variant_3_fcn_train.txt', need_feature_only=True)
    # infer_variant_3_fcn('test_java', 'features/feature_variant_3_fcn_test_java.txt', need_feature_only=True)
    # infer_variant_3_fcn('test_python', 'features/feature_variant_3_fcn_test_python.txt', need_feature_only=True)

    # print("Inferring variant 7 FCN...")
    # infer_variant_7_fcn('train', 'features/feature_variant_7_fcn_train.txt', need_feature_only=True)
    # infer_variant_7_fcn('test_java', 'features/feature_variant_7_fcn_test_java.txt', need_feature_only=True)
    # infer_variant_7_fcn('test_python', 'features/feature_variant_7_fcn_test_python.txt', need_feature_only=True)