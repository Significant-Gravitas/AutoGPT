import { useState } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Services from './components/Services';
import Sectors from './components/Sectors';
import WhyNeus from './components/WhyNeus';
import ContactForm from './components/ContactForm';
import DiagnosticForm from './components/DiagnosticForm';
import ChatWidget from './components/Chatbot/ChatWidget';
import Footer from './components/Footer';

function App() {
  const [isDiagnosticOpen, setIsDiagnosticOpen] = useState(false);

  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <Hero onOpenDiagnostic={() => setIsDiagnosticOpen(true)} />
      <Services />
      <Sectors />
      <WhyNeus />
      <ContactForm />
      <Footer />

      {/* Chat Widget */}
      <ChatWidget />

      {/* Diagnostic Modal */}
      <DiagnosticForm
        isOpen={isDiagnosticOpen}
        onClose={() => setIsDiagnosticOpen(false)}
      />
    </div>
  );
}

export default App;
