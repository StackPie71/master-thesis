import os
import shutil

def movefiles(dataroot):
    for name in os.listdir(dataroot):
        fileroot = dataroot+name
        print(fileroot)
        odpath = fileroot+'/'+filename
        osegpath = fileroot+'/'+segname
        newfilename = name + filename 
        newsegname = newfilename.replace('ta', 'ta_seg')
        ndpath = newroot+'/'+ newfilename
        nsegpath = segroot+'/'+ newsegname
        shutil.copy(odpath,ndpath)
        shutil.copy(osegpath,nsegpath)

if __name__ == '__main__':
    dataroot = '/auto/home/users/n/b/nboulang/Head-Neck-GTV-master/HeadNeck_GTV/alldataset/origindata/'
    saveroot = '/auto/home/users/n/b/nboulang/Head-Neck-GTV-master/HeadNeck_GTV/alldataset/origin_move/'
    newroot = saveroot+'data'
    segroot = saveroot+'label'
    filename = 'data.nii.gz'
    segname = 'label.nii.gz'
    movefiles(dataroot)