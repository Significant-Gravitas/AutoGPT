import { useState, FormEvent } from 'react';
import { Calendar, CheckCircle2, AlertCircle, Loader2, X } from 'lucide-react';
import { createAppointment } from '../services/api';
import { SECTORES, SERVICIOS } from '../types';

interface DiagnosticFormProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function DiagnosticForm({ isOpen, onClose }: DiagnosticFormProps) {
  const [formData, setFormData] = useState({
    nombre: '',
    email: '',
    empresa: '',
    sector: '',
    fecha_preferida: '',
    servicio_interes: '',
    mensaje: ''
  });

  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setStatus('loading');
    setErrorMessage('');

    try {
      await createAppointment({
        nombre: formData.nombre,
        email: formData.email,
        fecha_preferida: formData.fecha_preferida,
        ...(formData.empresa && { empresa: formData.empresa }),
        ...(formData.sector && { sector: formData.sector }),
        ...(formData.servicio_interes && { servicio_interes: formData.servicio_interes }),
        ...(formData.mensaje && { mensaje: formData.mensaje })
      });

      setStatus('success');
      // Reset form
      setFormData({
        nombre: '',
        email: '',
        empresa: '',
        sector: '',
        fecha_preferida: '',
        servicio_interes: '',
        mensaje: ''
      });

      // Close modal after 3 seconds
      setTimeout(() => {
        setStatus('idle');
        onClose();
      }, 3000);
    } catch (error) {
      setStatus('error');
      setErrorMessage(error instanceof Error ? error.message : 'Error al agendar el diagnóstico');
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-gradient-to-r from-primary-600 to-secondary-600 p-6 rounded-t-2xl flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Calendar className="h-6 w-6 text-white" />
            <h2 className="text-2xl font-bold text-white">
              Diagnóstico Gratuito
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/20 rounded-lg transition-colors"
          >
            <X className="h-6 w-6 text-white" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-8 space-y-6">
          <p className="text-gray-600">
            Agenda una sesión gratuita de diagnóstico y descubre cómo podemos ayudarte a reducir costos
            y aumentar la eficiencia de tu negocio.
          </p>

          {/* Nombre */}
          <div>
            <label htmlFor="diagnostic-nombre" className="block text-sm font-semibold text-gray-700 mb-2">
              Nombre <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              id="diagnostic-nombre"
              name="nombre"
              required
              value={formData.nombre}
              onChange={handleChange}
              className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-primary-600 focus:outline-none transition-colors"
              placeholder="Tu nombre completo"
            />
          </div>

          {/* Email */}
          <div>
            <label htmlFor="diagnostic-email" className="block text-sm font-semibold text-gray-700 mb-2">
              Email <span className="text-red-500">*</span>
            </label>
            <input
              type="email"
              id="diagnostic-email"
              name="email"
              required
              value={formData.email}
              onChange={handleChange}
              className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-primary-600 focus:outline-none transition-colors"
              placeholder="tu@email.com"
            />
          </div>

          {/* Empresa */}
          <div>
            <label htmlFor="diagnostic-empresa" className="block text-sm font-semibold text-gray-700 mb-2">
              Empresa
            </label>
            <input
              type="text"
              id="diagnostic-empresa"
              name="empresa"
              value={formData.empresa}
              onChange={handleChange}
              className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-primary-600 focus:outline-none transition-colors"
              placeholder="Nombre de tu empresa"
            />
          </div>

          {/* Sector */}
          <div>
            <label htmlFor="diagnostic-sector" className="block text-sm font-semibold text-gray-700 mb-2">
              Sector
            </label>
            <select
              id="diagnostic-sector"
              name="sector"
              value={formData.sector}
              onChange={handleChange}
              className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-primary-600 focus:outline-none transition-colors bg-white"
            >
              <option value="">Selecciona un sector</option>
              {SECTORES.map(sector => (
                <option key={sector} value={sector}>{sector}</option>
              ))}
            </select>
          </div>

          {/* Fecha Preferida */}
          <div>
            <label htmlFor="diagnostic-fecha" className="block text-sm font-semibold text-gray-700 mb-2">
              Fecha Preferida <span className="text-red-500">*</span>
            </label>
            <input
              type="date"
              id="diagnostic-fecha"
              name="fecha_preferida"
              required
              value={formData.fecha_preferida}
              onChange={handleChange}
              min={new Date().toISOString().split('T')[0]}
              className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-primary-600 focus:outline-none transition-colors"
            />
          </div>

          {/* Servicio de Interés */}
          <div>
            <label htmlFor="diagnostic-servicio" className="block text-sm font-semibold text-gray-700 mb-2">
              Servicio de Interés
            </label>
            <select
              id="diagnostic-servicio"
              name="servicio_interes"
              value={formData.servicio_interes}
              onChange={handleChange}
              className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-primary-600 focus:outline-none transition-colors bg-white"
            >
              <option value="">Selecciona un servicio</option>
              {SERVICIOS.map(servicio => (
                <option key={servicio} value={servicio}>{servicio}</option>
              ))}
            </select>
          </div>

          {/* Mensaje */}
          <div>
            <label htmlFor="diagnostic-mensaje" className="block text-sm font-semibold text-gray-700 mb-2">
              Mensaje
            </label>
            <textarea
              id="diagnostic-mensaje"
              name="mensaje"
              rows={3}
              value={formData.mensaje}
              onChange={handleChange}
              className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-primary-600 focus:outline-none transition-colors resize-none"
              placeholder="Cuéntanos brevemente sobre tu situación actual..."
            />
          </div>

          {/* Status Messages */}
          {status === 'success' && (
            <div className="flex items-center space-x-2 p-4 bg-green-50 border-2 border-green-500 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-green-500 flex-shrink-0" />
              <span className="text-green-700 font-medium">
                ¡Diagnóstico agendado! Te enviaremos un email de confirmación.
              </span>
            </div>
          )}

          {status === 'error' && (
            <div className="flex items-center space-x-2 p-4 bg-red-50 border-2 border-red-500 rounded-lg">
              <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
              <span className="text-red-700 font-medium">
                {errorMessage}
              </span>
            </div>
          )}

          {/* Submit Button */}
          <div className="flex space-x-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-6 py-3 bg-gray-200 text-gray-700 rounded-lg font-semibold hover:bg-gray-300 transition-colors"
            >
              Cancelar
            </button>
            <button
              type="submit"
              disabled={status === 'loading'}
              className="flex-1 flex items-center justify-center space-x-2 px-6 py-3 bg-gradient-to-r from-primary-600 to-secondary-600 text-white rounded-lg font-semibold hover:shadow-lg hover:scale-105 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
            >
              {status === 'loading' ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Agendando...</span>
                </>
              ) : (
                <>
                  <Calendar className="h-5 w-5" />
                  <span>Agendar Diagnóstico</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
