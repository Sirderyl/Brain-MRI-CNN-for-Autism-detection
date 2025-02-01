# This function extracts the pial and thickness data from dHCP subjects into
# one table on a hemisphere-basis. No smooth pial calculation. Everything based on the convexhull
#
# The code is heavily inspired on
# 
# https://github.com/cnnp-lab/CorticalFoldingAnalysisTools/tree/master
# 
#
# Original Authors
# Yujiang Wang, September 2016 (extractMaster_Hemisphere.m)
# Tobias Ludwig & Yujiang Wang, September 2019
# Newcastle University, School of Computing, CNNP Lab (www.cnnp-lab.com)
#
# Victor B. B. Mello, March 2024 
# Support Center for Advanced Neuroimaging (SCAN)
# University Institute of Diagnostic and Interventional Neuroradiology
# University of Bern, Inselspital, Bern University Hospital, Bern, Switzerland.
#
# For the Newcastle+Rio+Bern (NewcarRiBe) collaboration

import numpy as np
import nibabel as nib
import pymeshlab
import trimesh
import argparse
import os
import sys
import csv
from multiprocessing.pool import Pool

def partArea(vertices, faces, condition):
    # Area of a selected region
    # Dealing with the border: 1/3 triangle area if only one vertex is inside the ROI
    # 2/3 triangle area if 2 vertices are inside the ROI
    r_faces = np.isin(faces, condition)    
    fid = np.sum(r_faces,axis=1)
    area = 0
    array_area = trimesh.triangles.area(vertices[faces])
    
    for i in range(1,4,1):
        area = area + i*np.sum(array_area[fid==i])/3

    return area

def do_convex_hull(vertices, faces):
    ms = pymeshlab.MeshSet()            
    m = pymeshlab.Mesh(vertex_matrix=vertices,face_matrix=faces)
    ms.add_mesh(m,'surface')
    ms.generate_convex_hull()    
    ms.set_current_mesh(1)   

    return ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()
    
def transfer_label(initialv, initialf, scalar_label, finalv, finalf):
    # Transfer labels from mesh to mesh by proximity
    # label with the same label of the closest labeled vertex
    ms_transfer = pymeshlab.MeshSet()
    morig = pymeshlab.Mesh(vertex_matrix=initialv,face_matrix=initialf,v_scalar_array=scalar_label)
    mfinal = pymeshlab.Mesh(vertex_matrix=finalv,face_matrix=finalf)
    ms_transfer.add_mesh(morig,'initial_mesh')
    ms_transfer.add_mesh(mfinal,'final_mesh')        
    ms_transfer.transfer_attributes_per_vertex(sourcemesh = 0, targetmesh = 1, qualitytransfer = True)
    ms_transfer.compute_scalar_transfer_vertex_to_face()        
    ms_transfer.set_current_mesh(1)
    m = ms_transfer.current_mesh()
    
    return m.vertex_scalar_array()

def extract_FSHemi_features(input_list):
    # unpack input
    filepath = input_list[0]
    args = input_list[1]    
    csv_output = args.outputpath
    
    hemisphere = ["right","left"]
    ses_ID = filepath.split("/")[-1]
    subj_ID = filepath.split("/")[-2]
    file_ID = subj_ID+'_'+ses_ID
    
    result = []
        
    for hemi in hemisphere:
        error = False
        data_dict = {}
        # Reading dHCP reconstruction
        try:
            # Thickness
            thickness = nib.load(filepath+"/anat/"+file_ID+"_hemi-"+hemi+"_desc-corr_thickness.shape.gii").agg_data()                    
            # Pial surface
            pialv, pialf = nib.load(filepath+"/anat/"+file_ID+"_hemi-"+hemi+"_pial.surf.gii").agg_data()
          
            # Pure morphometrics from the FreeSurfer Meshes
            # units: mm
            # pial surface
            array_pialarea = trimesh.triangles.area(pialv[pialf])        
            pial_total_area = np.sum(array_pialarea)
            pial_volume = trimesh.triangles.mass_properties(pialv[pialf])['volume']

            # convex hull of the pial surface
            pialv_ch, pialf_ch = do_convex_hull(pialv,pialf)
            pial_convexhull_area = np.sum(trimesh.triangles.area(pialv_ch[pialf_ch]))                
            pial_convexhull_volume = trimesh.triangles.mass_properties(pialv_ch[pialf_ch])['volume']

            # Find the cc, brain stem ... to subtract the areas
            cc_pial_area = partArea(pialv, pialf, condition = np.where(thickness==0))
            
            # mean thickness
            avg_thickness = np.mean(thickness[thickness>0])

            # output data
            data_dict['subj'] = subj_ID            
            data_dict['ses'] = ses_ID
            data_dict['hemi'] = hemi
            data_dict['region'] = 'hemisphere'
            data_dict['total_pial_area'] = pial_total_area
            data_dict['total_pial_area_noCC'] = pial_total_area - cc_pial_area
            data_dict['total_pial_volume'] = pial_volume
            data_dict['total_convexhull_pial_area'] = pial_convexhull_area
            data_dict['total_convexhull_pial_area_noCC'] = pial_convexhull_area - cc_pial_area
            data_dict['total_convexhull_pial_volume'] = pial_convexhull_volume
            data_dict['thickness'] = avg_thickness
            data_dict['K'] = 0.5*np.log10(avg_thickness) + np.log10(pial_total_area) - 5*np.log10(pial_convexhull_area)/4
            data_dict['K_corrected'] = 0.5*np.log10(avg_thickness) + np.log10(pial_total_area - cc_pial_area) -5*np.log10(pial_convexhull_area - cc_pial_area)/4
            data_dict['I'] = 2*np.log10(avg_thickness) + np.log10(pial_total_area) + np.log10(pial_convexhull_area)
            data_dict['I_corrected'] = 2*np.log10(avg_thickness) + np.log10(pial_total_area - cc_pial_area) + np.log10(pial_convexhull_area - cc_pial_area)
            data_dict['S'] = -9*np.log10(avg_thickness)/2 + 3*np.log10(pial_total_area)/2 + 3*np.log10(pial_convexhull_area)/4
            data_dict['S_corrected'] = -9*np.log10(avg_thickness)/2 + 3*np.log10(pial_total_area - cc_pial_area)/2 +3*np.log10(pial_convexhull_area - cc_pial_area)/4            
            result.append(data_dict)

        except FileNotFoundError:
            print("WARNING: Missing reconstruction file from subject {}".format(file_ID), flush = True)
            error = True
            break
                                   
        except KeyboardInterrupt:
            print("The programm was terminated manually!")
            raise SystemExit                                       

    if error == False:
        file_exists = os.path.isfile(csv_output)
        with open (csv_output, 'a') as csvfile:
            headers = list(data_dict.keys())
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
        
            if not file_exists:
                writer.writeheader()
            writer.writerows(result)                        
                        
if __name__ == '__main__':
    # arguments from the shell: path to FS rec and the outputfile desired
    parser = argparse.ArgumentParser(prog='extract_dHCP_subj_hemisphere', description='This code extracts from a dHCP reconstruction cortical morphological variables in a hemisphere base for 1 subject')
    parser.add_argument('-filepath', '--path', help='path to the subject folder', required = True)
    parser.add_argument('-outputfile', '--outputpath', default = 'morphometric_data.csv', help='Path to the csv output file. Default is morphometric_data.csv to be saved at the running directory', required = False)
    args = parser.parse_args()

    # list of subjects packed together with args inside a tuple for the multiprocessing Map
    files_path = args.path
    ses_list = os.listdir(files_path)
    input_list = [(files_path +'/'+i, args) for i in ses_list]

    # create a log file for the corrupted IDs
    log_file = open("logfile.log","w")
    sys.stdout = log_file    
    
    # proccess multiple subjects at the same time
    with Pool() as pool:
        res = pool.map(extract_FSHemi_features, input_list)
 
    log_file.close()
