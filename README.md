# Battle Snake

This is a Python program for Solo Battle Snake.

You should follow this link to build your battlesnake on Replit and replace the code in main.py with this code.

[https://github.com/BattlesnakeOfficial/starter-snake-python](https://github.com/BattlesnakeOfficial/starter-snake-python)

## Solo Rules:

1. Solo
2. 7x7 board (this code is able to run on different sizes of the board)
3. Start with 100 health, decreasing by 1 with each move. Health regenerates to 100 when your snake eats food.
4. Always keep three food items on the board.
5. Food will only be created when there are fewer than three food items. (This code cannot run on the rule of randomly created food, or you should limit the amount of calculation to food to avoid exceeding the time limit.)
6. Decisions must be made within 500 milliseconds.

## Why I Created This:

In my college's PBL (Problem-Based Learning) class, we were tasked with creating a group Battle Snake program, and this is the idea I came up with.

## Algorithm:

### A\* Algorithm:

Finds the path to the tail and the path to the food.

## Strategy:

- When the snake's health is sufficient, it will prioritize searching for its tail and prefers routes without food.
- When the snake's health is insufficient, it will explore all available paths and select the one that leads to food with the shortest distance and is closest to maintaining health. It also checks if it's safe to eat and after ate the food.

## Result:

The program consistently achieves a score of around 3700 ~ 3800. The best score is 4100+. There may be some situations that make the snake dead at around 1500 ~ 2000.
