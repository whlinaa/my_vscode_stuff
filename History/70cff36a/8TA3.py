# %% [markdown]
# # function annotation example

# %%
from ds_preamble import *

# %%
def func(a: 'spam', b: (1, 10), c: float) -> int:
    print(a, b, c)
    print("hello")

func(1,2,3)


# %%
x: str = "hello"
x


# %%
from typing import List, Dict, Optional, Any, Sequence, Tuple,  Callable # note the capital letter!!
# y: list[list[int]] = []

dataframe = pd.DataFrame
array = np.ndarray


# y: np.ndarray[int] =[1,2,3]

# [[1,2,3],[4,5,6]]
y: List[List[int]] = [] # version <3.9
y: list[list[int]] = [] # version >=3.9
y: dict[str,str] = {'a':'b'}
y: Dict[str,str] = {'a':'b'}





print(y)


y: pd.DataFrame = [1,2,3]
y: pd.Series = [1,2,3]
y: list =[1,2,3]
y:list[dataframe]=[5.6]
 



def f1(y: Optional[dataframe] = None) -> int:
    """_summary_

    Args:
        y (Optional[dataframe], optional): _description_. Defaults to None.

    Returns:
        int: _description_
    """
    pass

def f2(y: dataframe = None) -> int:
    """_summary_

    Args:
        y (dataframe, optional): _description_. Defaults to None.

    Returns:
        int: _description_
    """
    pass


def f3(y: tuple[int]=(1,2,3,4,5)):
    pass


def f4(y: list[int]=[1,2,3,4,5]):
    pass

def f5(y: list[int, int]=[1,2,3,4,5]):
    pass



# %% [markdown]
# 


