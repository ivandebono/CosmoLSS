import camb
from camb import model, initialpower
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


class PKEvolution:

    def __init__(self):
        # Set up parameters
        self.pars = camb.CAMBparams()

        As = 2e-9  # fiducial amplitude guess to start with
        # Include massive neutrinos
        self.pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, 
                        omk=0, tau=0.06, neutrino_hierarchy='normal')


        self.pars.set_matter_power(redshifts=[0.0], kmax=4.0)
        self.results = camb.get_results(self.pars)
        
        s8_fid = self.results.get_sigma8_0()
        # now set correct As using As \propto sigma8**2.
        sigma8 = 0.81  # value we want
        self.pars.InitPower.set_params(As=As * sigma8**2 / s8_fid**2, ns=0.965,r=0)
  
        
        self.pars.WantTransfer = True
        self.pars.Transfer.redshifts = list(np.linspace(0,5,100))
        self.pars.Transfer.kmax = 5.0
        self.pars.Transfer.want_cl_transfer = False  # skip C_l transfer functions
        self.pars.Transfer.include_massive_neutrinos = True  # required to get neutrino TF

        # Set the redshift range and k range for P(k,z)
        self.pars.set_matter_power(redshifts=np.linspace(0,5,100),kmax=5.0,nonlinear=True,accurate_massive_neutrino_transfers=True)
        self.results = camb.get_results(self.pars)



    def mk_results(self):

        # Calculate results
        self.results = camb.get_results(self.pars)

        return self

    def mk_pkz(self):
        # Calculate matter power spectrum 
        self.kh,self.z,self.pk=self.results.get_matter_power_spectrum()

        return self
    
    def plt_pk(self,z,grid=True,figsize=(10,6)):
        z_index=z_index = np.where(np.isclose(self.z, z))[0][0]  # Find index of z

        fig,ax=plt.subplots(figsize=figsize)
        # Plot
        plt.loglog(self.kh, self.pk[z_index])
        plt.xlabel(r"$k$ [$h$ Mpc$^{-1}$]")
        plt.ylabel(r"$P(k)$ [$h^{-3}$ Mpc$^3$]")
        plt.title(f'Matter Power Spectrum at $z = {z}$')
        if grid:
            plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        plt.show()

        return fig


    def pk_animation(self, fps=10, filename=None,dpi=150,figsize=(10,6),grid=True,display=True):
        """Animates P(k) over z, with optional fixed axes."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set fixed axes limits (with 10% padding)
        x_min, x_max = np.min(self.kh), np.max(self.kh)
        y_min, y_max = np.min(self.pk), np.max(self.pk)
        padding = 0.1
        ax.set_xlim(x_min * (1 - padding), x_max * (1 + padding))
        ax.set_ylim(y_min * (1 - padding), y_max * (1 + padding))
        # Pre-compute z indices and axis limits (if fixed). We reverse the list so the animation goes forward in time,
        # i.e. a smaller redshigft is more recent
        z_indices = [np.where(np.isclose(self.z, z))[0][0] for z in self.z]
        z_indices.reverse()
        

        # Pre-initialize plot elements
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r"$k$ [$h$ Mpc$^{-1}$]")
        ax.set_ylabel(r"$P(k)$ [$h^{-3}$ Mpc$^3$]")
        if grid:
            ax.grid(True, which="both", ls="--", alpha=0.3)

            # Create empty line and title
        line, = ax.plot([], [], lw=2)  # Empty line
        text = ax.text(0.95, 0.95, '', transform=ax.transAxes,
               fontsize=12, ha='right', va='top')

        title="Non-linear matter power spectrum evolution"
        ax.set_title(title)
        
        
        def init():
            """Initialize animation (blank line)."""
            line.set_data([], [])
            text.set_text('')
            return line, text

    
        def update(frame):
            """Update line data and title for each z."""
            z_index = z_indices[frame]
            z=self.z[z_index]
            
            # Update line data (no axis rescaling)
            line.set_data(self.kh, self.pk[z_index])

            # Update text
            text.set_text(f"Redshift: $z$ = {z:.2f}")
        
            
            return line, text
        
        # Create animation
        ani = FuncAnimation(
        fig, update, frames=len(self.z),
        init_func=init, blit=True, interval=1000/fps)
    
    # Output handling
        if filename:
            ani.save(filename, writer='ffmpeg', fps=fps, dpi=dpi)
            plt.close(fig)
            return f"Animation saved to {filename}"
        if display:
            plt.close(fig)
            return HTML(ani.to_jshtml())
        else:
            return ani

