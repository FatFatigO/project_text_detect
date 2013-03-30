
function proj_0327_MSRA(order)

addpath_for_me;

% Project path init
PATH.FEATVEC_MAT = '../../_output_files/Feature_vectors/';
PATH.CLASSIFIER_MAT = '../../_output_files/Classifier/';
PATH.ADA_POST_MAT = '../../../../../LargeFiles/Adaboost_post/';
PATH.ADA_PRUNE_OUT_FOLDER = '../../../../../LargeFiles/Adaboost_prune_out/';
PATH.SVM_PRUNE_OUT_FOLDER = '../../../../../LargeFiles/Svm_prune_out/';
PATH.TRAINIMG_NONCHAR = '../../../../../LargeFiles/TrainImg_nonchar/';
PATH.TEST_FROM_PRUNED_ADA1 = '../../../../../LargeFiles/MSRATD500/[001] IMG_0059.JPG_400x400_ER_candidate_img_0_[0228_1555]/'; 
PATH.TEST_FROM_PRUNED_ADA2 = '../../../../../LargeFiles/MSRATD500/[004] IMG_0156.JPG_400x400_ER_candidate_img_0_[0228_1558]/';
PATH.TEST_FROM_PRUNED_ADA3 = '../../../../../LargeFiles/MSRATD500/[006] IMG_0172.JPG_400x400_ER_candidate_img_0_[0228_1111]/';
PATH.TEST_FROM_PRUNED_ADA4 = '../../../../../LargeFiles/MSRATD500/[012] IMG_0475.JPG_400x400_ER_candidate_img_0_[0228_1540]/';
PATH.TEST_FROM_PRUNED_ADA5 = '../../../../../LargeFiles/MSRATD500/[019] IMG_0507.JPG_400x400_ER_candidate_img_0_[0228_1526]/';
PATH.TEST_FROM_PRUNED_ADA6 = '../../../../../LargeFiles/MSRATD500/[034] IMG_0638.JPG_400x400_ER_candidate_img_0_[0228_1136]/';
PATH.TEST_FROM_PRUNED_ADA7 = '../../../../../LargeFiles/MSRATD500/[037] IMG_0667.JPG_400x400_ER_candidate_img_0_[0228_1140]/';

NAME.FEATVEC_MAT = '0311_test02_traditional_4_plus_3';
NAME.CLASSIFIER_MAT = '2ndStage_svm_20130218';
NAME.TESTING_DATASET = 'MSRATD500';
NAME.TESTING_SIZE = '400x400';

FEAT.RESIZE = [128 128];
FEAT.SHAPECONTEXT = [80 12 5 1/8 2 0];
FEAT.RANDPROJBITS = 8;
FEAT.RANDPROJMATRIX = normrnd(0,1,FEAT.SHAPECONTEXT(2)*FEAT.SHAPECONTEXT(3),FEAT.RANDPROJBITS);

STAGE = [ ...
              %     <Char74K (chars)>
    '001';... % r1: collect & save feat vectors from dataset
              %     <MSRATD500 (nonchars)>
    '110';... % r1: random and save as .png
              % r2: collect feat vectors from .png
              % r3: save feat vectors
              %     <Training SVM>
    '010';... % r1: train one versus all(w/o non) classifier for each chars
              % r2: train one versus all(w/  non) classifier for each chars
              % r3: train char versus nonchar classifier
              %     <Testing SVM>
    '010'];   % r1: test by Char74K images 
              % r2: test by Pruned Ada .png

% extract char features
% textdetect_a1_train_svm_Chars74K_all_vs_nonchars(PATH, NAME, FEAT, STAGE);

RULES.MIN_W_ABS = 3;
RULES.MIN_H_ABS = 3;
RULES.MIN_SIZE = 30;
RULES.MIN_W_REG2IMG_RATIO = 0.0019;
RULES.MAX_W_REG2IMG_RATIO = 0.4562;
RULES.MIN_H_REG2IMG_RATIO = 0.0100;
RULES.MAX_H_REG2IMG_RATIO = 0.7989;
RULES.PROB_MIN = 0.2;
RULES.DELTA_MIN = 0.1;
RULES.MIN_CONSEQ_ER_LEVEL = 2;
RULES.MAX_AREA_VARIATION = 0.05;

ds_eng = [];
ds_eng = imdataset('init', 'MSRATD500', ds_eng);
ds_eng = imdataset('get_test_dataset_deftxt_word', 'MSRATD500', ds_eng);
% seq = [6 7 9 19 34 37 55 63 76 78 88 89 98 129 131 139 141 142 149 151 152 153 159 165 190];
useSVM = 1;

if order(1)=='F' && order(5)=='P'
    seq = max(round(ds_eng.no*str2double(order(2:4))/100),1):1:ds_eng.no;
end
if order(1)=='B' && order(5)=='P'
    seq = min(round(ds_eng.no*str2double(order(2:4))/100),ds_eng.no):-1:1;
end
if order(1)=='F' && order(5)=='N'
    seq = str2double(order(2:4)):1:ds_eng.no;
end
if order(1)=='B' && order(5)=='N'
    seq = str2double(order(2:4)):-1:1;
end

for i=seq

    NAME.TESTING_IMG = util_changeFn(ds_eng.fn_list{i}, 'get_filename_and_extension', '');
    NAME.TESTING_IMG_IDX = i;
    ['[' num2str(i) ']' NAME.TESTING_IMG]

    text_detect_a4_classifyAdaPostp_optSVM_algo7_simple(PATH, NAME, FEAT, RULES, useSVM);
    
end



end