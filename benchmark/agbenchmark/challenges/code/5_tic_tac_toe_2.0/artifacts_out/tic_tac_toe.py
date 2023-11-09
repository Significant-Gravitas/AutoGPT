import pprint


def column(matrix, i):
    return [row[i] for row in matrix]


def check(list):
    if len(set(list)) == 1:
        if list[0] != 0:
            return list[0]
    return None


def checkDiagLeft(board):
    if board[0][0] == board[1][1] == board[2][2] == board[3][3]:
        if board[0][0] != 0:
            return board[0][0]
    return None


def checkDiagRight(board):
    if board[3][0] == board[2][1] == board[1][2] == board[0][3]:
        if board[3][0] != 0:
            return board[3][0]
    return None


def placeItem(row, column, board, current_player):
    if board[row][column] != 0:
        return None
    else:
        board[row][column] = current_player


def swapPlayers(player):
    if player == 3:
        return 1
    else:
        return player + 1


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
        "Choose where to play. Enter two numbers separated by a comma, for example: 1,1: "
    )
    print(f"\nYou picked {location}")
    coordinates = [int(x) for x in location.split(",")]
    while (
        len(coordinates) != 2
        or coordinates[0] < 0
        or coordinates[0] > 3
        or coordinates[1] < 0
        or coordinates[1] > 3
    ):
        print("You inputted a location in an invalid format.")
        location = input(
            "Choose where to play. Enter two numbers separated by a comma, for example: 1,1: "
        )
        coordinates = [int(x) for x in location.split(",")]
    return coordinates


def gamePlay():
    num_moves = 0
    pp = pprint.PrettyPrinter(width=30)
    current_player = 1
    board = [[0 for x in range(4)] for x in range(4)]

    while num_moves < 16 and winner(board) == 0:
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
