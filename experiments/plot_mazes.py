import matplotlib.pyplot as plt
import gym_maze
import numpy as np
import gymnasium as gym

import sys,os
os.chdir("..")
import utils

#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo a2c --env MiniGrid-DoorKey-8x8-v0 --frames 100000 --wrapper ssp-view --input flat --plot True --procs 1 --frames-per-proc 100 ', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
#
envs = np.array([ 'maze-sample-5x5-v0', 'maze-sample-6x6-v0', 'maze-sample-7x7-v0',
        'maze-sample-8x8-v0', 'maze-sample-9x9',
        'maze-sample-10x10', 'maze-sample-15x15', 'maze-sample-20x20',
        'maze-sample-25x25','maze-sample-50x50','maze-sample-100x100'])
envs = np.array([ 'maze-sample-10x10-v0'])

def write_svg(maze, filename):
    """Write an SVG image of the maze to filename."""

    aspect_ratio = maze.MAZE_W / maze.MAZE_H
    # Pad the maze all around by this amount.
    padding = 10
    # Height and width of the maze image (excluding padding), in pixels
    height = 500
    width = int(height * aspect_ratio)
    # Scaling factors mapping maze coordinates to image coordinates
    scy, scx = height / maze.MAZE_H, width / maze.MAZE_W

    def write_wall(ww_f, ww_x1, ww_y1, ww_x2, ww_y2):
        """Write a single wall to the SVG image file handle f."""

        print('<line x1="{}" y1="{}" x2="{}" y2="{}"/>'
              .format(ww_x1, ww_y1, ww_x2, ww_y2), file=ww_f)

    # Write the SVG image file for maze
    with open(filename, 'w') as f:
        # SVG preamble and styles.
        print('<?xml version="1.0" encoding="utf-8"?>', file=f)
        print('<svg xmlns="http://www.w3.org/2000/svg"', file=f)
        print('    xmlns:xlink="http://www.w3.org/1999/xlink"', file=f)
        print('    width="{:d}" height="{:d}" viewBox="{} {} {} {}">'
              .format(width + 2 * padding, height + 2 * padding,
                      -padding, -padding, width + 2 * padding, height + 2 * padding),
              file=f)
        print('<defs>\n<style type="text/css"><![CDATA[', file=f)
        print('line {', file=f)
        print('    stroke: #000000;\n    stroke-linecap: square;', file=f)
        print('    stroke-width: 5;\n}', file=f)
        print(']]></style>\n</defs>', file=f)
        # Draw the "South" and "East" walls of each cell, if present (these
        # are the "North" and "West" walls of a neighbouring cell in
        # general, of course).
        all_cells = [[{'N': 1, 'E': 1, 'S': 1, 'W': 1} for _ in range(maze.MAZE_H) ] for _ in range(maze.MAZE_W)]
        for x in range(maze.MAZE_W):
            for y in range(maze.MAZE_H):
                cell = maze.get_walls_status(maze.maze_cells[x, y])
                if cell['N']:
                    all_cells[x][y]['N'] = 0
                    if y>0:
                        all_cells[x][y-1]['S'] = 0
                if cell['S']:
                    all_cells[x][y]['S'] = 0
                    if y<maze.MAZE_H-1:
                        all_cells[x][y+1]['N'] = 0
                if cell['E']:
                    all_cells[x][y]['E'] = 0
                    if x<maze.MAZE_W-1:
                        all_cells[x+1][y]['W'] = 0
                if cell['W']:
                    all_cells[x][y]['W'] = 0
                    if x>0:
                        all_cells[x-1][y]['E'] = 0
            
        for x in range(maze.MAZE_W):
            for y in range(maze.MAZE_H):
                walls = all_cells[x][y]
                dx = x * scx
                dy = y * scy
                if walls['N']:
                    x1, y1, x2, y2 = dx + 1, dy, dx + scx -1, dy
                    write_wall(f, x1, y1, x2, y2)
                if walls['S']:
                    x1, y1, x2, y2 = dx + 1, dy + scy,dx + scx - 1, dy + scy  
                    write_wall(f, x1, y1, x2, y2)
                if walls['W']:
                    x1, y1, x2, y2 = dx, dy + 1, dx, dy + scy - 1
                    write_wall(f, x1, y1, x2, y2)
                if walls['E']:
                    x1, y1, x2, y2 = dx + scx, dy + 1,dx + scx, dy + scy - 1
                    write_wall(f, x1, y1, x2, y2)
                # if walls['S']:
                #     x1, y1, x2, y2 = x * scx, (y + 1) * scy, (x + 1) * scx, (y + 1) * scy
                #     write_wall(f, x1, y1, x2, y2)
                # if walls['E']:
                #     x1, y1, x2, y2 = (x + 1) * scx, y * scy, (x + 1) * scx, (y + 1) * scy
                #     write_wall(f, x1, y1, x2, y2)
                # if walls['W']:
                #     x1, y1, x2, y2 = x * scx, y * scy, x * scx, (y + 1) * scy
                #     write_wall(f, x1, y1, x2, y2)
        # Draw the North and West maze border, which won't have been drawn
        # by the procedure above.
        print('<line x1="0" y1="0" x2="{}" y2="0"/>'.format(width), file=f)
        print('<line x1="0" y1="0" x2="0" y2="{}"/>'.format(height), file=f)
        print('</svg>', file=f)

for i, env in enumerate(envs):
    gymenv = gym.make(env)#,enable_render=True)
    gymenv.reset()
    # gymenv.render()
    write_svg(gymenv.maze_view.maze, 'figures/' + env + '.svg')
   
