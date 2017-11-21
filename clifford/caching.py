"""
caching  

generates and caches algebras to a temp dir using pickle
"""

# delayed import to solve circularity
# from . import Cl

import pickle
import os
from os.path import join
from types import ModuleType
import tempfile

def get_temp_dir():
    return join(tempfile.gettempdir(),'clifford/')

def clear_cache():
    tmp_dir = get_temp_dir()
    filelist = [ f for f in os.listdir(tmp_dir) ]
    for f in filelist:
        os.remove(os.path.join(tmp_dir, f))
    
def generate_all_layout_combos(p_max=3, q_max=3, r_max=0,**kw):
    from . import Cl # delayed import 
    layouts = {}

    for p in range(p_max+1):
        for q in range(q_max+1):
            for r in range(r_max+1):
                if p+q<=1:
                    continue

                key = 'g%i%i%i'%(p,q,r)
                layouts[key],blades = Cl(p=p, q=q, **kw)
    return layouts 
    
def sigs_2_layouts(sigs):
    from . import Cl # delayed import 
    layouts ={}
    for p,q,r in sigs:
        layouts['g%i%i%i'%(p,q,r)] = Cl(p=p,q=q)[0]
    return layouts
    
def write_layouts(layouts, tmp_dir=None):
    if tmp_dir is None:
        tmp_dir = get_temp_dir()
    
    try:
        os.mkdir(tmp_dir)
    except(FileExistsError):
        pass
    
    for key in layouts:
        with open(join(tmp_dir, key+'.p'),'wb') as f:
            pickle.dump(layouts[key],f)
            print('caching %s'%key)
            
def read_layouts(tmp_dir=None):            
    if tmp_dir is None:
        tmp_dir = get_temp_dir()
    
    layouts = {}
    for fname in os.listdir(tmp_dir):
        
        with open(join(tmp_dir, fname),'rb') as f:
            key = os.path.splitext(fname)[0]
            layouts[key] = pickle.load(f)
           
    return layouts 

def generate_layout_submodules(layouts):
    #TODO: putting the whole cache into memory is dumb. need dynamic loading.
    submods = {}
    for key in layouts:
        # create layout submodule
        this_layout = layouts[key]
        mod = ModuleType(key)
        mod.__dict__.update({'layout':this_layout,
                            'blades':this_layout.blades})
        mod.__dict__.update(this_layout.blades)
        submods[key]=mod
    return submods
    
def build_or_read_cache_and_attach_submods(clif_mod,sigs=None):
    tmp_dir = get_temp_dir()
    cache_exists=False
    try:
        if len(os.listdir(tmp_dir))>0:
            cache_exists=True # cache exists
        else:
            pass # cache dir is empty
    except(FileNotFoundError):
        try:
            os.mkdir(tmp_dir)
        except(FileExistsError):
            pass
    
    if cache_exists is False:
        if sigs is None:
            layouts = generate_all_layout_combos() 
        else:
            layouts = sigs_2_layouts(sigs)
        write_layouts(layouts)

    # submods should be attached as requested, or all at once. 
    layouts = read_layouts()
    submods = generate_layout_submodules(layouts)

    # attach all algebra instances to clifford ,
    clif_mod.__dict__.update(submods)
