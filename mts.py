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
import cPickle
import itertools
import IPython
import img_util

SSIZE = 2

class Texel:
    """ Represents stack level Texel """
    def __init__(self, exemplar, level):
        self.exemplar = exemplar
        self.level = level
        self.admissibles = []
        self.kcache = {}
    def image(self):
        """return the stack image"""
        return self.exemplar.stack[self.level]

class Exemplar:
    """ Represents Image Exemplar """
    def __init__(self, image, id):
        """creates gaussian stack and exemplars"""
        self.stack = img_util.gaussian_stack(image) #stacks of level 0, 1, ... L - 1
        self.stack_size = len(self.stack) #aka L+1 where image is m x m, m = 2^L.
        self.texels = [Texel(self, level) for level in range(self.stack_size)]
        for texel in self.texels:
            if texel.level < self.stack_size - 1:
                texel.admissibles.append(texel)
        self.relations = []
        self.id = id
    def make_relation(self, other_exemplar, scale):
        """creates relation with another exemplar, where the other exemplar is the target"""
        self.relations.append((other_exemplar, scale))
        for source_texel in self.texels: 
            target_level = source_texel.level - scale
            if target_level >= 0 and target_level != other_exemplar.stack_size - 2:
                source_texel.admissibles.append(other_exemplar.texels[target_level])
    def __repr__(self):
        return "(exemplar : id : {}   stack size : {}       stack (first shape) : {}      num texels : {}     relations : {})".format(self.id, self.stack_size, self.stack[0].shape, len(self.texels), [(rel[0].id, rel[1]) for rel in self.relations])

#i, k, u (x,y) : which exemplar, which stack level, what coordinate.
#r, g, b: what color (for fast rendering)
#7D array?
class RenderPyramid:
    """ Data structure that holds synthesis data """
    def __init__(self, exemplars):
        self.data_pyramid = []
        self.exemplars = exemplars
        #self.color_pyramid = []
        self.next_render_level = 0
    def data_to_color(self, level):
        data_dimension = self.data_pyramid[level].shape[0]
        color_image = np.zeros((data_dimension, data_dimension, 3), dtype = np.int16)
        for x, y in itertools.product(range(data_dimension), range(data_dimension)):
            i, k, ux, uy = self.data_pyramid[level][x][y]
            color_image[x][y] = self.exemplars[i].stack[k+1][ux][uy]
        return np.clip(color_image, 0, 255).astype(np.uint8)
        #return color_image
    
    def correct_next_level(self):
        color_image = self.data_to_color(self.next_render_level - 1)
        IWIDTH = color_image.shape[0]
        for sx, sy in itertools.product(range(IWIDTH), range(IWIDTH)):
            data_matrix = self.data_pyramid[self.next_render_level - 1]
            i, k, ux, uy = data_matrix[sx][sy]
            L = self.exemplars[i].stack_size -1
            SWIDTH = SSIZE*2 + 1
            reference = np.zeros((SWIDTH, SWIDTH, 3), dtype=np.uint8)
            for dx, dy in itertools.product(range(-SSIZE, SSIZE+1), range(-SSIZE, SSIZE+1)):
                reference[dx + SSIZE][dy + SSIZE] = color_image[(sx+dx)%IWIDTH][(sy+dy)%IWIDTH]
            this_texel = self.exemplars[i].texels[k]
            if len(this_texel.admissibles) == 0:
                if k == L:
                    return False
                continue
            min_norm = float('inf')
            argmin = None
            for admissible in this_texel.admissibles:
                downsample = max(2**(L - admissible.level - 1), 1)
                search_img = admissible.image()#[::downsample, ::downsample, :]
                si_width = search_img.shape[0]
                #pad image
                search_img = np.dstack((np.pad(search_img[:,:,0], (SSIZE*downsample,), 'wrap'), 
                                            np.pad(search_img[:, :, 1], (SSIZE*downsample,), 'wrap'),
                                                np.pad(search_img[:, :, 2], (SSIZE*downsample,), 'wrap')))
                
                for tx, ty in itertools.product(range(si_width), range(si_width)):
                    search_patch = search_img[tx:tx+(2*SSIZE+1)*downsample:downsample, 
                                                    ty:ty+(2*SSIZE+1)*downsample:downsample, :]
                    norm = np.linalg.norm(reference - search_patch)
                    if norm < min_norm:
                        min_norm = norm
                        argmin = admissible.exemplar.id, admissible.level, tx, ty
                #reconstruct actual coordinates from argmin
                texel_width = admissible.image().shape[0]
            if argmin is None:
                continue
            #argmin = argmin[0], argmin[1], (argmin[2] * downsample + sx % downsample) % texel_width, (argmin[3] * downsample + sy % downsample) % texel_width
            data_matrix[sx][sy] = argmin
        return True
                
    def supersample_level(self):
        source_dimension = self.data_pyramid[self.next_render_level - 1].shape[0]
        new_data_matrix = np.zeros((source_dimension * 2, source_dimension * 2, 4), dtype = np.int16)
        for sx, sy in itertools.product(range(source_dimension), range(source_dimension)):
            data_matrix = self.data_pyramid[self.next_render_level - 1]
            i, k, ux, uy = data_matrix[sx][sy]
            img_dimension = self.exemplars[i].stack[0].shape[0]
            L = self.exemplars[i].stack_size - 1
            h = 2**(L-k-2)
            for dx, dy in itertools.product((0, 1), (0, 1)):
                new_data_matrix[2*sx+dx][2*sy+dy] = i, k + 1, (ux + h*dx - h/2) % img_dimension, (uy + h * dy - h/2) % img_dimension
        self.data_pyramid.append(new_data_matrix)
        self.next_render_level += 1
    def root_init(self):
        assert(self.next_render_level == 0)
        dimension = self.exemplars[0].stack[0].shape[0]
        image = np.zeros((1, 1, 4), dtype = np.int16)
        image[0][0] = 0, 0, dimension / 2, dimension / 2
        self.data_pyramid.append(image)
        self.next_render_level = 1
    def get_render(self):
        return self.data_to_color(self.next_render_level - 1)
        
class RenderCore:
    """ Object that holds everything needed to render and display """
    def __init__(self, exemplars, render_data):
        self.exemplars = exemplars
        self.render_data = render_data
        self.render_data.root_init()

    def debug_display(self):
        # temporary
        image = self.render_data.get_render()
        image = np.clip(image, 0, 255).astype(np.uint8)
        cv2.imshow("image", image)
        cv2.waitKey(0)
    def debug_save(self):
        #temporary
        image = self.render_data.get_render()
        image = np.clip(image, 0, 255).astype(np.uint8)
        cv2.imwrite("core_debug_save.png", image)
        
    def render_next(self):
        self.render_data.supersample_level()
        status =self.render_data.correct_next_level()
        assert(status)

def cache_calculate(exemplar):
    width = exemplar.texels[0].image().shape[0]
    for texel in exemplar.texels:
        texel_image = texel.image()
        for admissible in texel.admissibles:
            if admissible is texel:
                for sx, sy in itertools.product(range(width), range(width)):
                    if(sx, sy) not in texel.kcache:
                        texel.kcache[(sx, sy)] = []
                    texel.kcache[(sx, sy)].append((exemplar.id, texel.level, sx, sy))
                continue
            L = exemplar.stack_size - 1
            downsample = max(2**(L - admissible.level - 1), 1)
            search_img = admissible.image()#[::downsample, ::downsample, :]
            si_width = search_img.shape[0]
            #pad image
            search_img = np.dstack((np.pad(search_img[:,:,0], (SSIZE*downsample,), 'wrap'), 
                                        np.pad(search_img[:, :, 1], (SSIZE*downsample,), 'wrap'),
                                            np.pad(search_img[:, :, 2], (SSIZE*downsample,), 'wrap')))
            #vectorize target
            target_vectors = np.zeros((width*width, (2*SSIZE+1)*(2*SSIZE+1)*3))
            for tx, ty in itertools.product(range(width), range(width)):
                search_patch = search_img[tx:tx+(2*SSIZE+1)*downsample:downsample, 
                                                    ty:ty+(2*SSIZE+1)*downsample:downsample, :]
                target_vectors[tx+ty*width] = search_patch.flatten()
            #pad source image
            source_img = np.dstack((np.pad(texel_image[:,:,0], (SSIZE,), 'wrap'), 
                                            np.pad(texel_image[:, :, 1], (SSIZE,), 'wrap'),
                                                np.pad(texel_image[:, :, 2], (SSIZE,), 'wrap')))
            speed = 10
            for sx, sy in itertools.product(range(0, width, speed), range(0, width, speed)):
                source_patch = source_img[sx:sx+(2*SSIZE+1), sy:sy+(2*SSIZE+1), :]
                source_vector = source_patch.flatten()
                distance_vectors = target_vectors - source_vector
                distance_vectors = distance_vectors * distance_vectors
                distance_vectors = distance_vectors.sum(1)
                index = distance_vectors.argmin()
                y = index / width
                x = index - y*width
                if (sx, sy) not in texel.kcache:
                    texel.kcache[(sx, sy)] = []
                texel.kcache[(sx, sy)].append((admissible.exemplar.id, admissible.level, x, y))
                if sx == sy:
                    print(sx)
            for sx, sy in itertools.product(range(width), range(width)):
                if sx % speed == 0 and sy % speed == 0:
                    continue
                else:
                    if (sx, sy) not in texel.kcache:
                        texel.kcache[(sx, sy)] = []
                    nearest = texel.kcache[((sx / speed)*speed, (sy/speed)*speed)][-1]
                    texel.kcache[(sx, sy)].append((nearest[0], nearest[1], nearest[2] + (sx % speed), nearest[3] + (sy % speed)))
            
                
            
def setup_render(source_dir):
    """
    source_dir should be of the form "path/path/"
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
    for exemplar in exemplars:
        cache_calculate(exemplar)
    return exemplars
def debug_show(image):
    cv2.imshow("debug", np.clip(image, 0, 255).astype(np.uint8))
    cv2.waitKey(0)
def debug_save(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("debug_save.png", np.clip(image, 0, 255).astype(np.uint8))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-source", action="store", help="source image directory")
    parser.add_argument("-render", action="store", help="render directory")
    args = vars(parser.parse_args())
    #testing stuff!
    """
    exemplars = setup_render(args["source"])
    dump_file = open("render/rock/exemplars.p", 'wb')
    cPickle.dump(exemplars, dump_file)
    assert(False)
    """
    # delete this stuff later
    exemplars = None
    with open("render/rock/exemplars.p", 'rb') as file:
        exemplars = cPickle.load(file)
    pyramid = RenderPyramid(exemplars)
    core = RenderCore(exemplars, pyramid)
    core.render_next()
    core.render_next()
    core.render_next()
    core.render_next()
    core.render_next()
    #core.render_next()
    #core.render_next()
    #core.render_next()
    IPython.embed()
    core.debug_save()

    #debug_save(exemplars[0].stack[9])