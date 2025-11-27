#
import numpy as np
import os
import argparse
import xml.etree.ElementTree as ET
from PIL import Image

def change_field(root, field_name, value):
    """
    Change the view normal vector in the XML object.
    """
    for obj in root.iter('Field'):
        if obj.get('name') == field_name:
            print(f"Processing object: {obj.get('name')} value: {obj.text}")
            obj.text = value
            print(f"Updated object: {obj.get('name')} to value: {obj.text}")


def change_view_normal(root, x, y, z):
    """
    Change the view normal vector in the XML object.
    """
    for obj in root.iter('Field'):
        if obj.get('name') == "viewNormal":
            print(f"Processing object: {obj.get('name')} value: {obj.text}")
            obj.text = f"{x} {y} {z}"
            print(f"Updated object: {obj.get('name')} to value: {obj.text}")

def change_view_up(root, x, y, z):
    """
    Change the view up vector in the XML object.
    """
    for obj in root.iter('Field'):
        if obj.get('name') == "viewUp":
            print(f"Processing object: {obj.get('name')} value: {obj.text}")
            obj.text = f"{x} {y} {z}"
            print(f"Updated object: {obj.get('name')} to value: {obj.text}")

def change_output_directory(root, out):
    """
    Change the output directory in the XML object.
    """
    for obj in root.iter('Field'):
        if obj.get('name') == "outputDirectory":
            print(f"Processing object: {obj.get('name')} value: {obj.text}")
            obj.text = f"{out}"
            print(f"Updated object: {obj.get('name')} to value: {obj.text}")
            
    
def change_file_name(root, out, newname):
    """
    Change the file name in the XML object.
    """
    for obj in root.iter('Field'):
        if obj.get('name') == "lastRealFilename":
            print(f"Processing object: {obj.get('name')} value: {obj.text}")
            obj.text = f"{out}/{newname}"
            print(f"Updated object: {obj.get('name')} to value: {obj.text}")
        elif obj.get('name') == "fileName":
            obj.text = newname


def experiment(file_base, nrotate, out, begin=0, end=360, znorm=0.2, uznorm=0.9):
    """
    Run the experiment with the given arguments.
    """
    print(f"Running experiment with args: {args}")
    # Here you would implement the logic to run your experiment
    # For example, you might call a function that processes MRI data
    # or runs a simulation based on the provided options.
    end_point = True if end > begin else False
    deg_angles = np.linspace(begin, end, nrotate, endpoint=end_point)

    # Read in the file
    with open(file_base, 'r') as file :
        filedata = file.read()
    # Parse the XML file
    tree = ET.parse(file_base)
    root = tree.getroot()

    change_output_directory(root, out)


    for i, deg_ang in enumerate(deg_angles):
        
        ang = np.deg2rad(deg_ang)+0.01
        
        z = znorm #+ 0.2 * np.sin(ang)**2
        r = np.sqrt(1-z**2)

        x=r*np.cos(ang)
        y=r*np.sin(ang)
        
        # Change the normal vector from the XML
        change_view_normal(root, x, y, z)

        
        

        # Change the up vector from the XML
        z=uznorm
        x=np.sqrt(1-z**2)*np.cos(ang)
        y=np.sqrt(1-z**2)*np.sin(ang)

        change_view_up(root, x, y, z)
        

        newname = f"angle_{i:03}.png"

        change_file_name(root, out, newname)
        

        change_field(root, "imageZoom", str(1.0))

        change_field(root,"imagePan","-0.0574576939756343 0.0532909332051073")

        change_field(root,"parallelScale","130")
        

        #filedata = filedata.replace('angle_000', f'350_angle_{i:03}')

        fname='rotation'+str(i)+'.session'
        tree.write(fname, xml_declaration=True)
        # Write the file out again
        
        visit_bin = "/usr/local/visit/bin/visit"
        command=(f'{visit_bin} -cli -nowin -s restore_print.py {fname}')
        os.system(command)


        # # Opens a image in RGB mode
        # im = Image.open(f"{out}/{newname}")

        # # Setting the points for cropped image
        # left = 225#0
        # top = 200#2*64
        # right = 1750#2000
        # bottom = 2*883

        # # Cropped image of above dimension
        # # (It will not change original image)
        # im1 = im.crop((left, top, right, bottom))

        # # save as pdf
        # pdfname = f"{out}/crop_angle_{i:03}.pdf"
        # print(f"Saving cropped image as PDF: {pdfname}")
        # im1.save(os.path.join(out, pdfname), "PDF", resolution=100.0)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(exit_on_error=True, description='Reconstruct network')
    parser.add_argument('--base', type=str, help='Base filename for the session files.', required=True)
    parser.add_argument('--n', type=int, help='Number of rotations to perform.', required=True)
    parser.add_argument('--begin', type=float, default=0.0, help='Starting angle in degrees.')
    parser.add_argument('--end', type=float, default=360.0, help='Ending angle in degrees.')
    parser.add_argument('--znorm', type=float, default=0.02, help='Z normalization factor.')
    parser.add_argument('--uznorm', type=float, default=0.99, help='UZ normalization factor.')
    parser.add_argument('--out', type=str, default="./plots", help='output directory for the plots.')
    args, unknown = parser.parse_known_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    experiment(args.base, nrotate=args.n, out=os.path.abspath(args.out), 
               begin=args.begin, end=args.end, 
               znorm=args.znorm, uznorm=args.uznorm)
    
    print("Experiment completed successfully.")
