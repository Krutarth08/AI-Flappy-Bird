import pygame
import random
import time
import neat

win_width = 500
win_height = 600

pygame.font.init()
bg = pygame.transform.scale(pygame.image.load("images/bg.png"),(win_width,win_height))
bird = [pygame.image.load("images/bird1.png"),pygame.image.load("images/bird2.png"),pygame.image.load("images/bird3.png")]
pipe_image = pygame.transform.scale2x(pygame.image.load("images/pipe.png"))
base_img = pygame.transform.scale2x(pygame.image.load("images/base.png"))
Font = pygame.font.SysFont('comicsans', 40, bold=True, italic=False)

#Class

class Bird:
    img = bird
    flapper = 5
    flapper_count = 0
    current_img = img[0]
    speed = 0

    last_jump = 0
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.height = y

    def jump(self):
        self.speed = -10.5
        self.last_jump = 0 
        self.height = self.y

    def draw(self,win):
        self.flapper_count+=1
        if(self.flapper_count <= self.flapper) : 
            self.current_img = self.img[0]
        
        elif(self.flapper_count <= self.flapper*2) : 
            self.current_img = self.img[1]
        
        elif(self.flapper_count <= self.flapper*3) : 
            self.current_img = self.img[2]
        
        elif(self.flapper_count <= self.flapper*4) :
             self.current_img = self.img[1]

        elif(self.flapper_count <= self.flapper*4 + 1) : 
            self.current_img = self.img[0]
            self.flapper_count = 0

        win.blit(self.current_img,(self.x,self.y))

    def move(self):
        self.last_jump+=1
        temp_distance = self.last_jump*self.speed  +   1.5*self.last_jump**2
        #201.5,207.5

        if(temp_distance > 16) : temp_distance = 16       
        self.y += temp_distance

class Pipe :
    GAP = 200
   
    velocity = 4
    def __init__(self,x):
         self.x = x
         self.pipe_up = pygame.transform.flip(pipe_image,False,True)
         self.pipe_down = pipe_image
         self.connector = 0
         self.top = 0
         self.bottom = 0
         self.set_position()
         self.ispassed = False
     
    def set_position(self):    
         self.connector = random.randrange(50,400)
         self.top = self.connector - self.pipe_up.get_height()
         self.bottom = self.connector+self.GAP

    def draw(self,win):
         win.blit(self.pipe_up,(self.x,self.top))
         win.blit(self.pipe_down,(self.x,self.bottom))

    def move(self):
        self.x -= self.velocity

    def collision(self,bird):
        bird_mask = pygame.mask.from_surface(bird.current_img)
        top_mask = pygame.mask.from_surface(self.pipe_up)
        bottom_mask = pygame.mask.from_surface(self.pipe_down)

        top_offset = (int(self.x - bird.x), int(self.top - bird.y))
        bottom_offset = (int(self.x - bird.x), int(self.bottom - bird.y))

        top_result = bird_mask.overlap(top_mask,top_offset)
        bottom_result = bird_mask.overlap(bottom_mask,bottom_offset)

        if(top_result or bottom_result):
            return True
        return False           

class Base :
    speed = 2
    img = base_img
    width = base_img.get_width()
    def __init__(self,y):
        self.y = y
        self.x1 = 0 
        self.x2 = self.width

    def move(self):
        self.x1-=self.speed
        self.x2-=self.speed   

        if(self.x1 + self.width < 0):
            self.x1 = self.x2 + self.width
        if(self.x2 + self.width < 0):
            self.x2 = self.x1 + self.width

    def draw(self,win):
        win.blit(self.img,(self.x1,self.y))
        win.blit(self.img,(self.x2,self.y))
                             

#Main
def main(genomes,configfile):
    win = pygame.display.set_mode((win_width,win_height))
    bird_objs = []
    networks = []
    generations = []

    for __,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,configfile)
        networks.append(net)
        bird_objs.append(Bird(200,200))
        g.fitness = 0
        generations.append(g)


    pipe_obj = [Pipe(600)]
    base_obj = Base(600)
    flag = True
    tick_clock = pygame.time.Clock()
    pygame.init()
    score = 0
    while flag:
            tick_clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    flag = False    

            index_pipe = 0
            if len(bird_objs) > 0:
                if len(pipe_obj)>1 and bird_objs[0].x > pipe_obj[0].x + pipe_obj[0].pipe_up.get_width():
                    index_pipe = 1
            else:
                flag = False
                break    

            for x,bird in enumerate(bird_objs):
                bird.move()
                generations[x].fitness+=1

                output = networks[x].activate((bird.y, abs(bird.y - pipe_obj[index_pipe].connector), abs(bird.y - pipe_obj[index_pipe].bottom)))

                if output[0] > 0.5:
                    bird.jump()

            add_pipe = False

            remove_pipes = []

            for pipe in pipe_obj:
                for x,bird in enumerate(bird_objs):    
                    if pipe.collision(bird):
                        generations[x].fitness-=0.1
                        bird_objs.pop(x)
                        networks.pop(x)
                        generations.pop(x)
                        pass

                    if not pipe.ispassed and pipe.x<bird.x:
                        pipe.ispassed = True
                        add_pipe = True
                
                if pipe.x + pipe.pipe_up.get_width()<0:
                    remove_pipes.append(pipe)        
                pipe.move()
                base_obj.move()
            
            if(add_pipe):
                score+=1
                for g in generations:
                    g.fitness+=5
                pipe_obj.append(Pipe(500))
            for r in remove_pipes:
                pipe_obj.remove(r) 

            for x,bird in enumerate(bird_objs):
                if bird.y + bird.current_img.get_height() > 700 or bird.y < 0:
                    bird_objs.pop(x)
                    networks.pop(x)
                    generations.pop(x)    
                            
            win.blit(bg,(0,0))    #blit is for displaying something at particular the coordinates  
            for i in pipe_obj:
                i.draw(win)
            text = Font.render("Score:  " + str(score),1,(230,255,255))
            win.blit(text,(win_width - 10 - text.get_width(),10)) # 10 is y coordinate or height    
            base_obj.draw(win) 
            for bird in bird_objs:
                bird.draw(win)

            #bird_obj.move()
            pygame.display.update()


def run(configpath):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         configpath)
        populate = neat.Population(config)
	    
        winner = populate.run(main,50)

if __name__ == '__main__':
	configfile = "neatconfig.txt"
	run(configfile)   
