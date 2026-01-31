import pprint


def column(matrix, i):
    return [row[i] for row in matrix]


def check(list):
    if len(set(list)) <= 1:
        if list[0] != 0:
            return list[0]
    return None


def checkDiagLeft(board):
    if board[0][0] == board[1][1] and board[1][1] == board[2][2]:
        if board[0][0] != 0:
            return board[0][0]
    return None


def checkDiagRight(board):
    if board[2][0] == board[1][1] and board[1][1] == board[0][2]:
        if board[2][0] != 0:
            return board[2][0]
    return None


def placeItem(row, column, board, current_player):
    if board[row][column] != 0:
        return None
    else:
        board[row][column] = current_player


def swapPlayers(player):
    if player == 2:
        return 1
    else:
        return 2


def winner(board):
    for rowIndex in board:
        if check(rowIndex) is not None:
            return check(rowIndex)
    for columnIndex in range(len(board[0])):
        if check(column(board, columnIndex)) is not None:
            return check(column(board, columnIndex))
    if checkDiagLeft(board) is not None:
        return checkDiagLeft(board)
    if checkDiagRight(board) is not None:
        return checkDiagRight(board)
    return 0


def getLocation():
    location = input(
        "Choose where to play. Enter two numbers separated by a comma [example: 1,1]: "
    )
    print(f"\nYou picked {location}")
    coordinates = [int(x) for x in location.split(",")]
    while (
        len(coordinates) != 2
        or coordinates[0] < 0
        or coordinates[0] > 2
        or coordinates[1] < 0
        or coordinates[1] > 2
    ):
        print("You inputted a location in an invalid format")
        location = input(
            "Choose where to play. Enter two numbers separated by a comma "
            "[example: 1,1]: "
        )
        coordinates = [int(x) for x in location.split(",")]
    return coordinates


def gamePlay():
    num_moves = 0
    pp = pprint.PrettyPrinter(width=20)
    current_player = 1
    board = [[0 for x in range(3)] for x in range(3)]

    while num_moves < 9 and winner(board) == 0:
        print("This is the current board: ")
        pp.pprint(board)
        coordinates = getLocation()
        placeItem(coordinates[0], coordinates[1], board, current_player)
        current_player = swapPlayers(current_player)
        if winner(board) != 0:
            print(f"Player {winner(board)} won!")
        num_moves += 1

    if winner(board) == 0:
        print("Draw")


if __name__ == "__main__":
    gamePlay()
