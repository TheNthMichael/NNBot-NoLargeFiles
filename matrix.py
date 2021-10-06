from typing import List


def rotate(matrix: List[List[int]]) -> None:
    shelltl = (0,0)
    shelltr = (0,len(matrix)-1)
    shellbl = (len(matrix)-1, 0)
    shellbr = (len(matrix)-1, len(matrix)-1)
    copya = None
    copyb = None
    space = len(matrix)

    # order tl->tr->br->bl
    """
    Do not return anything, modify matrix in-place instead.
    """
    tag = 0
    while (shelltl != shelltr and shelltr != shellbl and shellbl != shellbr) and (shellbr[0] > shelltl[0]):
        for i in range(space - 1):
            # save top right
            copya = matrix[shelltr[0] + i][shelltr[1]]
            # top right equals top left
            matrix[shelltr[0] + i][shelltr[1]] = matrix[shelltl[0]][shelltl[1] + i]
            # save bottom right
            copyb = matrix[shellbr[0]][shellbr[1] - i]
            # bottom right equal top right (saved)
            matrix[shellbr[0]][shellbr[1] - i] = copya
            # move to original save loc
            copya = copyb
            # save bottom left
            copyb = matrix[shellbl[0] - i][shellbl[1]]
            # bottom left equals bottom right
            matrix[shellbl[0] - i][shellbl[1]] = copya
            # no need to save top left as its been moved
            # top left equals bottom left
            matrix[shelltl[0]][shelltl[1] + i] = copyb
            # all corners have been rotated
            copya = None
            copyb = None
        shelltl = (shelltl[0] + 1, shelltl[1] + 1)
        shelltr = (shelltr[0] + 1, shelltr[1] - 1)
        shellbr = (shellbr[0] - 1, shellbr[1] - 1)
        shellbl = (shellbl[0] - 1, shellbl[1] + 1)
        space -= 2
    print(matrix)
        

if __name__ == "__main__":
    rotate([[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]])