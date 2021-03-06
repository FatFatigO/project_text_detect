function text_detect_a4_classifyAdaPostp_optSVM_algo2(fd, fn, resize, classifier_fn_tag, rules, useSVM)

close all;
addpath_for_me;
tic

a = clock;
time_label = sprintf('[%02d%02d_%02d%02d]', a(2), a(3), a(4), a(5));
path = util_changeFn('','cd ..','');
path = util_changeFn(path,'cd ..','');
path = util_changeFn(path,'cd _mkdir','_output_files');
string_fr = [fn '_' num2str(resize(1)) 'x' num2str(resize(2))];
in_path = [path 'Parsed_mat/'];
out_path = util_changeFn([path 'Output_img/'],'cd _mkdir',[time_label ' ' string_fr '_ER_candidate_img' ]);
load(['../../_output_files/Classifier/2ndStage_svm_' classifier_fn_tag '.mat'], 'svm');

for reverse = 0:1

    load([in_path string_fr '_reverse_' num2str(reverse) '.mat']); 
    
    % find index of ER cadidates to be char region
    for row = 1:size(pmap,1)
        postp = pmap(row,:);
        idx = (postp>=rules.DELTA_MIN);
        idx = conv(double(idx), ones(1,rules.MIN_CONSEQ_ER_LEVEL));
        idx = (idx>=rules.MIN_CONSEQ_ER_LEVEL);
        isfirst = 1;
        for i=1:length(idx)
            if idx(i)==1
                if isfirst==1
                    isfirst = 0;
                    % first idx t of these consequtive ERs
                    t = i-rules.MIN_CONSEQ_ER_LEVEL+1; 
                    r = ft_ert.feat_raw.fmap(row,t);
                    % label is_done as 2 to indicate ER candidate
                    ft_ert.feat_raw.tree{t,r}.isdone = 2; 
                else
                    if idx(i+1)==0
                        % last idx t of these consequtive ERs
                        t = i;
                        r = ft_ert.feat_raw.fmap(row,t);
                        % label is_done as 2 to indicate ER candidate
                        ft_ert.feat_raw.tree{t,r}.isdone = 2; 
                    end
                end
            end
        end
    end

    % save ER candidates as images
    c1 = 0; c2 = 0;
    for t=1:255
        for r = 1:ft_ert.feat_raw.size(t)
            if ft_ert.feat_raw.tree{t,r}.isdone == 2
                % get ER data
                fst = ft_ert.feat_raw.tree{t,r}.raw(3);
                num = ft_ert.feat_raw.tree{t,r}.raw(2);
                vec = ft_ert.feat_raw.pxls(fst:fst+num-1)+1; % correct start index as Matlab sense
                TR_data = false(1, ft_ert.w*ft_ert.h);
                TR_data(vec) = 1;
                data = reshape(TR_data, ft_ert.w, ft_ert.h)'; % row-wised reshape
                c1 = c1 + 1;

                % do 2nd stage classification
                if useSVM
                    isChar = svmclassify(svm, ft_ert.feat_raw.tree{t,r}.feat_vec);
                    if ~isnan(isChar) && isChar
                        % save ER as image
                        s = [out_path 'ER_(' num2str(t) ',' num2str(r) ')_reverse_' num2str(reverse) '.png'];
                        imwrite(data, s, 'png')
                        c2 = c2 + 1;
                    end
                else
                    % save ER as image
                    s = [out_path 'ER_(' num2str(t) ',' num2str(r) ')_reverse_' num2str(reverse) '.png'];
                    imwrite(data, s, 'png')
                    c2 = c2 + 1;
                end
            end
        end
    end

    % save original image (for reference)
    total_ER_no = sum(ft_ert.feat_raw.size);
    s = [out_path '__[3]no_of_ER_(all,c1,c2)=(' num2str(total_ER_no) ',' num2str(c1) ',' num2str(c2) ')_reverse_' num2str(reverse) '.png'];
    if reverse == 0
        original_img = ft_ert.image;
    else
        original_img = 255 - ft_ert.image;
    end
    imwrite(original_img, s, 'png')

    % save accum image (for reference)
    fns = dir([out_path 'ER*_' num2str(reverse) '.png']);
    [H,W] = size(imread([out_path fns(1,1).name]));
    I_accum = false(H,W);
    for i=1:numel(fns)
        I = logical(imread([out_path fns(i,1).name])); 
        I_accum = I | I_accum;
    end
    s = [out_path '__[1]accum_' num2str(c2) 'ERs_reverse_' num2str(reverse) '.png'];
    imwrite(I_accum, s, 'png');
    
    % save normal MSER (for reference)
    if 0
    im = [];
    im = imfeat('init', 'mser', im);
    im = imfeat('set_image', original_img, im);
    im = imfeat('convert', '', im);
    mser_param = '';
    im = imfeat('extract_feature_raw', mser_param, im);
    set(figure, 'Position', [100, 100, im.w, im.h]);
    imagesc(im.image);
    axis off; hold on;
    pause(1);
    plot(im.feat_raw, 'showEllipses',false, 'showPixelList',true);
    [X, map] = frame2im(getframe(gca));
    s = [out_path '__[2]mser_image_reverse_' num2str(reverse) '.png'];
    imwrite(X, s, 'png')
    hold off;
    close(gcf);
    end

end
toc
end
