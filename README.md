# Battle Snake

This is a Python program for playing Battle Snake.

## Rules:

1. Solo play
2. 7x7 board
3. Start with 100 health, decreasing by 1 with each move. Health regenerates to 100 when your snake eats food.
4. Always keep three food items on the board.
5. Food will only be created when there are fewer than three food items.
6. Decisions must be made within 500ms, including internet latency.

## Why I Created This:

In my college's PBL (Problem-Based Learning) class, we were tasked with creating a group Battle Snake program, and this is the idea I came up with.

## Algorithm:

### A* Algorithm:
Finds the path to the tail and the path to the nearest food.

## Strategy:

- When the snake's health is sufficient, it will prioritize searching for its tail and prefers routes without food.
- When the snake's health is insufficient, it will explore all available paths and select the one that leads to food with the shortest distance and is closest to maintaining health. It also checks if it's safe to eat that food.

## Result:

The program consistently achieves a score of around 2500.
