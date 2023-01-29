import tkinter as tk
import numpy as np
from numpy.linalg import solve, norm, det
# ベクトルの正則化
re = lambda x: x / norm(x)

class Canvas(tk.Canvas):
    def __init__(self, parent, size=np.array([800,600]), dx=150*np.array([1,-1])):
        super().__init__(parent, width=size[0], height=size[1])
        self.size = size
        self.dx = dx
        self.tags = []
    def toR(self, *x): # resizeに対して不変な座標への変換
        return (x - self.size/2) / self.dx
    def toS(self, *x):
        return x * self.dx + self.size/2
root = tk.Tk()
canvas = Canvas(root)
canvas.pack(expand=True, fill=tk.BOTH)

class Ball():
    def __init__(self, subroom, dir, object):
        self.subrooms = [subroom]
        self.dirs = [dir]
        self.ts = [0]
        self.object = object
        self.converged = False
        self.predict_collision()
    
    def pos(self, t):
        x = self.subrooms[-2][0]
        v = self.dirs[-2]
        return x + v*t
    
    def predict_collision(self):
        x, v = self.subrooms[-1], self.dirs[-1]
        stlist = np.empty((x.shape[0]-2,2))
        for i in range(x.shape[0]-2):
            # x[0] + v*t = x[i+1]*(1-s) + x[i+2]*sを解く
            mat = np.array([x[i+1]-x[i+2],v]).T
            stlist[i] = solve(mat, x[i+1]-x[0]) if det(mat) != 0 else np.inf
        # 凹多角形の場合を考慮して0<=s<=1を条件に追加
        tlist = np.where((stlist[:,1]>=0) & (stlist[:,0]>=0) & (stlist[:,0]<=1), stlist[:,1], np.inf)
        col = tlist.argmin()
        t = tlist[col]
        if (np.abs(t) < 1e-8): # 収束判定
            self.converged = True
            self.subrooms.append(x)
            self.dirs[-1] = np.zeros(2)
            self.dirs.append(np.zeros(2))
            self.ts.append(np.inf)
            return
        
        e = re(x[col+2]-x[col+1])
        idx = range(col+2,x.shape[0]+1,1) if v.dot(e)>0 else range(col+1,-1,-1)
        next_v = 2*v.dot(e)*e - v
        next_x = np.array([x[0]+v*t] + [x[i%x.shape[0]] for i in idx])

        self.subrooms.append(next_x)
        self.dirs.append(next_v)
        self.ts.append(t)

ball = Ball(subroom=np.array([[-1,-1],[-1,1],[1,1],[1,-1]]), dir=re([2,3]), object=None)

dt = 1/30
t_sect = 0
# 描画の内容
def draw():
    points = np.array([canvas.toS(*x) for x in ball.subrooms[0]]).reshape(-1)
    if ('room' not in canvas.tags):
        canvas.create_polygon(*points, fill="", outline="black", tag='room')
        canvas.tags.append('room')
    else:
        canvas.coords('room', *points)
    points = np.tile(canvas.toS(*ball.pos(t_sect)),2) + np.concatenate([-canvas.dx, canvas.dx])*0.02
    if ('ball' not in canvas.tags):
        ball_obj = canvas.create_oval(*points, tag='ball', fill='black')
        canvas.tags.append('ball')
        ball.object = ball_obj
    else:
        canvas.coords('ball', *points)
    points = np.array([canvas.toS(*x[0]) for x in ball.subrooms[:-1]] + [canvas.toS(*ball.pos(t_sect))]).reshape(-1)
    if ('line' not in canvas.tags):
        canvas.create_line(*points, tag='line')
        canvas.tags.append('line')
    else:
        canvas.coords('line', *points)
    
# 実行
draw()

def on_resize(event):
    global canvas
    canvas.size = np.array([event.width, event.height])
    draw()

def auto_update():
    global t_sect
    draw()
    t_sect += dt
    if (t_sect >= ball.ts[-1]):
        t_sect = 0
        ball.predict_collision()
    root.after(int(1000*dt), auto_update)

def on_enter_press(event):
    auto_update()

root.bind_all("<Key-Return>", on_enter_press)

root.bind("<Configure>", on_resize)

root.mainloop()
