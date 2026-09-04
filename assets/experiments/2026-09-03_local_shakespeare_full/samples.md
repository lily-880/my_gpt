# greedy 采样摘录

来源：2026-09-03 本机训练日志。  
设置：prompt=`ROMEO:`，`max_new_tokens=40`，每步 `argmax`。未做 temperature / top-k。

空行是模型自己生成的，不是复制时删掉了字。

## step 200

```
ROMEO: 
 
CASSIUS. 
 
 [_Exit._] 
 
IUS. 
 
 [_Exit._]
```

## step 400

```
ROMEO: 
I’ll be not.
```

（后面大量空行）

## step 600

```
ROMEO:
And that the King’s eyes,
And in the world, and in the world,
And in the world, and in the world,
And in the world, and in
```

## step 800

```
ROMEO: 
And I will not be a man. 
 
 [_Exit._]
```

（后面大量空行）

## step 1000

```
ROMEO: 
I am glad to be a little man. 
 
 [_Exit._] 
 
SCENE III. The same. The same. 
 
 Enter Ant
```

## step 1200

```
ROMEO: 
I’ll not be a man. 
 
 [_Exeunt._] 
 
SCENE III. Another part of the Castle.
```

## step 1400

几乎全是空行。greedy 一旦抽到换行就会锁死，不能用来判断这一步的语言模型质量。

## step 1600

```
ROMEO: 
O, that thou hast done, and I have done 
The world, and the world, and the world, 
The world, and the world, and the world,
```

## step 1800

```
ROMEO: 
The sun is like a king, and his son, 
And in the world’s death, that in the world 
To make a king of a king.
```

这次最像「在写诗」，但仍有 world/king 套话。

## step 2000

```
ROMEO:
And she, nor she, nor she, nor she,
To be her, and her love her love.

PERICLES.
I will not love her love.
```

人物名和剧本换说话人的格式出现了；内容仍是复读。
