# first-day-tune-up

Unsure about anything? Ask us!

## Text Editor

First, get set up with a text editor.  If you don't already have a text editor you like, use [Sublime Text](http://www.sublimetext.com/2). It runs on Linux, Windows, and Mac OS X.

## Mac OS X

These instructions assume that you have OS X 10.10 installed. If you have an older version of Mac OS X, upgrading to 10.10 is a great idea because its free.

### If you need to install 10.10:

If you're on 10.6.6 or later: You can get 10.10 in the Mac App Store.

If you are running a version of 10.6 older than 10.6.6, you first need to run Software Update from System Preferences and upgrade to 10.6.8. After the update, the Mac App Store will appear in your dock and you can use it to install OS X 10.10.

If you are running OS X 10.5 or below, you will have to pay $20 to upgrade to 10.6 before being able to upgrade to 10.10.

If your computer is too old to run 10.10, don't worry - many of these instructions will work with older versions of Mac OS X as well. If you can afford it, you should upgrade to the latest version of OS X that supports your computer.

### Settings for your terminal and $PATH

Your terminal uses a environmental variable called the PATH to find programs. PATH is a list of directories to be searched for programs. You can see it by running `echo $PATH`.

If you have a file called `.bash_profile` (notice the leading period) in your home directory, move everything in it to a file called `.bashrc`. `.bashrc` is the script that runs when you open a new terminal window or tab, and it's often responsible for setting environmental variables that configure different tools. If you don't have a `.bash_profile` in your home directory, create one. Replace the contents of your .bash_profile with the following:

    if [ -f ~/.bashrc ]; then
       source ~/.bashrc
    fi

This means "If a file called `.bashrc` exists, run that script." It's useful because some programs look for .bash_profile instead of .bashrc, and you want to have all your config in one place.

After making changes to your `.bashrc` or `.bash_profile`, you need to close your terminal window. While it's possible to reload your settings in your existing terminal window, it can sometimes lead to complications.

### Install the Command Line Tools (C compiler, GNUMake, etc.)

If you're running OS X 10.9 or 10.10:

Run `xcode-select --install` in your terminal, and follow the instructions. This will install the Command Line Tools.  If you get an error like "Software not available from the download sever", you can download the Command Line Tools from [Apple's developer downloads site](https://developer.apple.com/download). You'll need an Apple ID to log in.

If you are planning on writing Mac or iOS apps, you can optionally install Xcode from the App Store. This is a large download.

If you're running 10.6 - 10.8 and can't upgrade to 10.9 or 10.10:

Install the latest version of Xcode from the App Store. Open Xcode, select the Xcode menu and choose Preferences..., select "Downloads" and install the latest version of the Command Line Tools. You may be asked to log in to your Apple developer account. If you don't have a developer account, it's free to register. It uses the same username and password that you use for the App Store.

If you're running 10.5 or earlier:

You have a really old version of OS X. Even if you can't upgrade to 10.10, you should try to upgrade to the latest version that supports your computer. This might cost some money. If you can't upgrade, go to Apple's [developer downloads site](https://developer.apple.com/downloads/) and download the latest version of Xcode that supports your OS. Wikipedia has a [list of supported versions](https://en.wikipedia.org/wiki/Xcode).

### Install [Homebrew](http://brew.sh/)

Run the command under "Install instructions" on [brew.sh](http://brew.sh/)
Run `brew update`
Run `brew doctor` and fix any problems it points out
Run `brew install git`
Run `brew install bash-completion`.  Once this is finished, brew will recommend you add a few lines of bash code to your bash_profile. (Look under "Caveats".) **Add it to your .bashrc instead of .bash_profile.**

### Additional Sublime Text setup

Once you've installed Sublime Text, set up the `subl` command to open files or directories from the command line by creating a symbolic link:
Sublime Text 2: `ln -s /Applications/Sublime\ Text\ 2.app/Contents/SharedSupport/bin/subl /usr/local/bin/subl` (if you're on Sublime Text 3: `ln -s "/Applications/Sublime Text.app/Contents/SharedSupport/bin/subl" /usr/local/bin/subl`) (/usr/local/bin should be writable because you installed homebrew. No need to sudo).

Add this line to your bashrc `export EDITOR='subl -w'`. This makes any program that wants a text editor (e.g. git) use Sublime.

#### Package Control

In addition to built in support for many languages, Sublime Text has support for community created plugins that let Sublime Text interact with other languages and tools. To make it easier to install these community plugins, you can [install Package Control](https://sublime.wbond.net/installation), a package manager for Sublime Text.

### Python on Mac OS X

If you're writing Python:

Do `which -a python` to see how many Pythons you have installed.

_If you see **only** `/usr/bin/python`_, then you're good to go.

_If not:_

Delete any non-homebrew non-system pythons (these will be pythons you installed from python.org). Important note: You don't want to remove your system Python. It comes with Mac OS X and is installed by Apple in /System/Library. If you haven't installed a Python from python.org, there is no need to uninstall anything. Don't delete anything in /System/Library!
`rm -rf /Applications/MacPython\ <VERSION>` (might be Python instead of MacPython)
Remove everything that comes up in `ls -al /usr/local/bin | grep /Library/Frameworks/Python.framework` for the version of Python you're uninstalling
`rm -rf /Library/Frameworks/Python.framework` This will kill all versions of python.org Python. You may want to only remove a specific one.
Look in your `.bash_profile` or `.bashrc` Remove anything that the python.org Python added to your `$PATH`

Now:
`brew install python` - includes pip, the python package manager
`pip install virtualenv`

## Linux (Debian-based including Ubuntu)

```
sudo apt-get install build-essential
sudo apt-get install git
sudo apt-get install bash-completion
```

### Python on Linux (Debian-based including Ubuntu)

```
sudo apt-get install python python-dev
wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py -O - | sudo python # this installs easy_install
sudo easy_install pip
sudo pip install virtualenv
```

## Linux (Fedora-based)

```
sudo yum groupinstall "Development Tools"
sudo yum install git # (I haven't verified this)
sudo yum install bash-completion # (I haven't verified this)
```

## Windows

### Git

Download and install [Git for Windows](https://msysgit.github.io/), then open the new-installed Git Bash and follow the Git config instructions below.

### Ruby

[RubyInstaller](http://rubyinstaller.org/) is a Windows-compatible version of Ruby.

### Python

You can follow [these instructions](http://docs.python-guide.org/en/latest/starting/install/win/) to install Python and setuptools/pip on Windows.

To update your PATH without powershell (so it is easier to run Python scripts from the command line), follow these steps:

1. From your desktop/the start menu, right-click on Computer and choose Properties.
2. Select "Advanced System Settings" in the dialog box, and then "Environment Variables".
3. Edit the PATH statement in the User Variables section to add:
```
C:\Python27;C:\Python27\Lib\site-packages\;C:\Python27\Scripts\;
```

### Alternate approach: use a virtual machine
Another option for working on windows is to use a virtual machine like [VirtualBox](https://www.virtualbox.org/) and install Linux on it. VirtualBox is an "emulator" that runs a fake Linux machine on top of your Windows machine. If you've never used Linux before, we recommend using [Ubuntu](http://www.ubuntu.com/download/desktop).

Once you've installed VirtualBox and a Linux distribution, you can follow the Linux instructions.

## Ruby (for Linux or OS X)
If you're planning to write Ruby, follow these installation instructions.
We use [RVM](https://rvm.io/), the Ruby Version Manager, to install Ruby.

Run this in your shell (the leading '\' is not a typo): `\curl -sSL https://get.rvm.io | bash -s stable --ignore-dotfiles --without-gems="rvm rubygems-bundler"`
Add the following lines to the end of your `.bashrc`. Remember to close all your terminal windows after saving your `.bashrc` file.

    if [[ -s $HOME/.rvm/scripts/rvm ]] ; then source $HOME/.rvm/scripts/rvm ; fi
    if [[ -s $HOME/.rvm/scripts/completion ]] ; then source $HOME/.rvm/scripts/completion ; fi

In a new terminal, run `type rvm | head -1`. It should output `rvm is a function`.
`rvm install ruby` This will install the latest version of ruby, currently 2.2.0, and set it as your default.
Don't use `sudo` with `gem install` even if you see instructions telling you to do so.

## Git

Git is a version control program. It allows you to take snapshots of your code and get back to them at any time. It also lets you collaborate with other programmers. Once you have installed git (see OS specific instructions above, you need to configure it. Here are some good default configuration options. You can type these into any terminal window:

```
git config --global user.name "Your Name Here"
git config --global user.email "your_email@example.com"
git config --global color.ui auto
```

You can have Bash prompt show what branch you're on if you're inside a git repository. It looks something like this: `~/Development/jeffmaxim [master]$ ` . You set your prompt by setting the `$PS1` environmental variable. You can look at [Jeff's friend Allison's](https://gist.github.com/akaptur/7b1340bc54e05e76ad4c) or [Jeff's friend Dave's](https://github.com/davidbalbert/dotfiles/blob/e029b0edeaa4d61ae4c876777bf1f9a3083736e5/bashrc#L45-L48) for examples and fiddle to your hearts content. These lines go in your `.bashrc` file. The `__git_ps1` function is responsible for printing the git branch in your prompt. The `$(...)` syntax calls the `__git_ps1` function and places its return value into your prompt. If you want to see your current prompt, you can run `echo $PS1`.

### Git credential helper

If you're using Git over HTTPS (as opposed to SSH), you will have to enter your GitHub username and password every time you pull from any private repository or push to any public or private repository. Entering your username and password every time gets annoying. To get around this, you can use a credential helper, which is a program that saves your username and password for you in a safe place.

On Mac OS X, this is pretty easy. You can use the `git-credential-osxkeychain` helper to store your username and password in Keychain, which is OS X's encrypted password store. To set this up, run this line in your terminal:

```
git config --global credential.helper osxkeychain
```

Linux is a bit more complicated because different distributions have different encrypted password stores. If you're using a GNOME based distribution, you can use `git-credential-gnome-keyring` which uses GNOME Keyring, another encrypted password store, to store your GitHub username and password.

Here's how you configure it on Ubuntu 13.10. This will probably work on other versions of Ubuntu and other Debian based Linux distributions. With some modifications (replacing `apt-get` with `yum` and figuring out the right package names and directories) this should work on Fedora-based Linux distributions running GNOME as well.

```
sudo apt-get install libgnome-keyring-dev
cd /usr/share/doc/git/contrib/credential/gnome-keyring
sudo make
git config --global credential.helper /usr/share/doc/git/contrib/credential/gnome-keyring/git-credential-gnome-keyring
```
