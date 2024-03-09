# tinyGPT
PyTorch implementation of GPT(Generative pretrained Transformer)


A thank you to Andrej Karpathy and his great <a href="https://youtu.be/kCc8FmEb1nY">lecture.</a>

Dependencies:

- pytorch

Sample of the generated text with a model trained on the tiny shakespeare dataset. The model was trained with the hyperparameters that are currently expressed in the main.py file.
At 3000 iterations the train loss is 1.205 and the validation loss is 1.512. After 3000 iterations the model starts overfitting and the validation loss starts diverging.

```
FRIAR PHARD S:
Our earth, then, fiend duked houring that men.
Iron by wits thoses orcary maiden kies all I.

RICHMOND:
It shall be hasted but the morning of it!
What may is nothing; and start I, and in heir
Three such virtue, and their slifest arm.

QUEEN MARGARET:
Moreatime, where should not, as you shall have,
In alive his began that doth power'd:
Nothing how it, to sand unnatural discent;
Northrown against that himself and pity itter.
In this deep-cents in death, to treads to the realm,
And with the peever garlant than I be sounded.
They naked of these gentle took in this soul,
Bad off. Have you, revenued my stood lies. All hold,
In lose banished such a child promott.

Nurse:
Murder; so withs Tree?

MERCUCUTIO:
What cress mild should none of my hand and hem hen;
But I'll have come into this envy's bowry.
```

At the 5000 iteration mark, the training loss goes down to ~0.989, but the validation loss goes up to ~1.624. Sample of the generated text:

```
LUCIO:
Gracious pardon, I hope, hark, you both.
Go ye nou subjects: therefore, I will perch you.
Think how you seem here, by smiles of my back.

POLIXENES:
I cannot withdraw you before comes  too fair.

Clown:
Come, come, come, we will hear ourself; all then have our in peace.

MERCUTIO:
I wish'd her fortunes and see his master. And what we given
To sin fell our drunks him that we learn'd her to exeach?

BRUTUS:
No money: good consentle to your for tellinght,
And therefore your vapond xilege for Calauce speak.

Roman:
Ray not 'er come; nay, do wish't you off the other
To see faith? So do, no, we would you, to save your
conspiect and I well warrant, as thou art not:
I speak out thy will talk in the world's wife,
Which for my affairs my scope to sundle my lord.

Second Citizen:
Cousin, your voices, sir, let the hand.

MENENIUS:
Well, he ha worthy Migorima.

First Citizen:
You, sir, my lord.

```