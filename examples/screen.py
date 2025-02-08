import pygame
import sys
import time

class HDMIDisplay:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        
        # Get the current screen info
        screen_info = pygame.display.Info()
        self.width = screen_info.current_w
        self.height = screen_info.current_h
        
        # Set up the display in full screen mode
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        pygame.display.set_caption("Full Screen Display")
        
        # Set up fonts - make it bigger for full screen
        self.font = pygame.font.Font(None, 72)  # Increased font size
        
        # Set up colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        
        # Display state
        self.display_state = 0
        self.last_update = time.time()
        
    def clear_screen(self):
        """Clear the screen to black"""
        self.screen.fill(self.BLACK)
        
    def display_text(self, text, secondary_text=None):
        """Display text on the screen"""
        self.clear_screen()
        
        # Render main text
        text_surface = self.font.render(text, True, self.WHITE)
        text_rect = text_surface.get_rect(center=(self.width/2, self.height/3))
        self.screen.blit(text_surface, text_rect)
        
        # Render secondary text if provided
        if secondary_text:
            secondary_surface = self.font.render(secondary_text, True, self.WHITE)
            secondary_rect = secondary_surface.get_rect(center=(self.width/2, 2*self.height/3))
            self.screen.blit(secondary_surface, secondary_rect)
        
        pygame.display.flip()
            
    def update_display(self):
        """Update the display based on state"""
        current_time = time.time()
        
        # Update every 2 seconds
        if current_time - self.last_update >= 2:
            if self.display_state == 0:
                self.display_text("Ari has a huge cock!")
                self.display_state = 1
            else:
                self.display_text("And Sam Altman wants it", "Ari has a huge cock!")
                self.display_state = 0
            self.last_update = current_time

    def run(self):
        """Main loop"""
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:  # ESC to quit
                        running = False
                    elif event.key == pygame.K_f:     # F to toggle fullscreen
                        pygame.display.toggle_fullscreen()
            
            # Update display
            self.update_display()
            
            # Small delay to prevent high CPU usage
            pygame.time.delay(100)

def main():
    # Create and run display
    display = HDMIDisplay()
    display.run()
    
    # Clean up
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
