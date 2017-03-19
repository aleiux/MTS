"""
(i, j, r) element of E, 
i = which source exemplar
j = which destination exemplar
r = similarity relation (2^r)

(i, k, u)
i = Which exemplar
j = Which stack level
u = Offset(?)

Where S0[p] = (i, k, u)
S1[2p + delta + (1/2, 1/2)] = i, k+1, u+ delta*2^(L-k)


S0 is one pixel. References E0
For level (Si) in image pyramid:
    For pixel in image pyramid: 
        Calculate i, k, u from relation
 
"""
import numpy as np
import argparse
import cv2
import re

import img_util


class Texel:
    """ Represents stack level Texel """
    def __init__(self, exemplar, level):
        self.exemplar = exemplar
        self.level = level
        self.admissibles = []
    def has_next(self):
        return self.level + 1 < self.stack_size
    def next(self):
        """get the next texel in the exemplar for super sampling"""
        return self.exemplar.texels[level + 1]
    def image(self):
        """return the stack image"""
        return self.exemplar.stack[level]
 
class Exemplar:
    """ Represents Image Exemplar """
    def __init__(self, image, id):
        """creates gaussian stack and exemplars"""
        self.stack = img_util.gaussian_stack(image) #stacks of level 0, 1, ... L - 1
        self.stack_size = len(self.stack) #aka L where image is m x m, m = 2^L.
        self.texels = [Texel(self, level) for level in range(self.stack_size)]
        self.relations = []
        self.id = id
    def make_relation(self, other_exemplar, scale):
        """creates relation with another exemplar, where the other exemplar is the target"""
        self.relations.append((other_exemplar, scale))
        for source_texel in self.texels: 
            target_level = source_texel.level - scale
            if target_level >= 0 and target_level != other_exemplar.stack_size - 1:
                source_texel.admissibles.append(other_exemplar.texels[target_level])
    def __repr__(self):
        return "(exemplar : id : {}   stack size : {}       stack (first shape) : {}      num texels : {}     relations : {})".format(self.id, self.stack_size, self.stack[0].shape, len(self.texels), [(rel[0].id, rel[1]) for rel in self.relations])

#i, k, u (x,y) : which exemplar, which stack level, what coordinate.
#r, g, b: what color (for fast rendering)
#7D array?
class RenderPyramid:
    """ The object that carries all the data in the rendering step """
    def __init__(self):
        self.data_pyramid = []
        self.color_pyramid = []
        
def setup_render(source_dir, render_dir):
    """
    source_dir and render_dir should be of the form "path/path/"
    Setup render environment. If metadata exists already, read that in.
    """
    relation_text = [line for line in open(source_dir + "relation.txt", 'r')] 
    root_name = relation_text[0].strip()
    exemplar_relations = [line.split() for line in relation_text[1:]]
    image_names = set([val for sublist in exemplar_relations for val in sublist if not str.isdigit(val)]) #looks complicated but just flattens the list keeping only the image names, then extracts the unique ones via set
    sorted_names = sorted(list(image_names))
    sorted_names.remove(root_name)
    sorted_names.insert(0, root_name)
    name_to_id = dict([(sorted_names[i], i) for i in range(len(sorted_names))])
    #this ensures that the exemplars always have the same order across runs. iterating through a set should always be consistent, but just in case...
    exemplars = [None] * len(sorted_names)
    for image_name in image_names:
        exemplars[name_to_id[image_name]] = Exemplar(cv2.imread(source_dir + image_name), name_to_id[image_name])
    for relation in exemplar_relations:
        source_exemplar = exemplars[name_to_id[relation[0]]]
        target_exemplar = exemplars[name_to_id[relation[1]]]
        source_exemplar.make_relation(target_exemplar, int(relation[2]))
    #Exemplar and Texel graph established
    return exemplars

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-source", action="store", help="source image directory")
    parser.add_argument("-render", action="store", help="render directory")
    args = vars(parser.parse_args())
    setup_render(args["source"], args["render"])