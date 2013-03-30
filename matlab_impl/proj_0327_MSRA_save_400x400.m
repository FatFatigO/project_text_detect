function proj_0327_MSRA_save_400x400()

% type:
%   1: output "word" region as small images
%   2: output "char" region as small images
%   3: output "non-char" region as small images

% dataset:
%   1: ICDAR2003RobustReading
%   2: MSRATD500

addpath_for_me;
folder_name = '400x400_original';
% dataset initialization
ds_eng = [];
ds_eng = imdataset('init', 'MSRATD500', ds_eng);
ds_eng = imdataset('get_train_dataset_deftxt_word', '', ds_eng);
ft_bin = [];
ft_bin = imfeat('init', 'binary', ft_bin);

for i=1:ds_eng.no
    % prepare for output image path
    file = util_changeFn(ds_eng.fn_list{i}, 'get_filename_and_extension', '');
    path = util_changeFn(ds_eng.fn_list{i}, 'remove_filename_and_extension', '');
    path = util_changeFn(path, 'cd ..', '');
    path = util_changeFn(path, 'cd _mkdir', '_output_files');
    path = util_changeFn(path, 'cd _mkdir', folder_name);
    path = [path sprintf('%03d', i) '.jpg'];
    
    img = imread(ds_eng.fn_list{i});
    cmd2 = [400, 400];
    param.h = size(img, 1);
    param.w = size(img, 2);
    % resize and keep aspect ratio
    rt_w = cmd2(1) / param.w; 
    rt_h = cmd2(2) / param.h;
    % smaller ratio will ensure that the image fits in the view
    if rt_w <= rt_h
        param.w = round(param.w * rt_w);
        param.h = round(param.h * rt_w);
    else
        param.w = round(param.w * rt_h);
        param.h = round(param.h * rt_h);        
    end
    param.image = imresize(img, [param.h param.w]);
    
    I = param.image;
    out_path = path;%util_changeFn(path, 'replace_extension', [num2str(ii) '.jpg']);
    imwrite(I, out_path, 'jpeg')

    
end

end