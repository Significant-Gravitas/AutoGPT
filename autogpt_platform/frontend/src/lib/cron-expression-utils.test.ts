import { makeCronExpression } from './cron-expression-utils';

describe('makeCronExpression', () => {
  it('should return empty string for yearly schedule with no months selected', () => {
    const result = makeCronExpression({
      frequency: 'yearly',
      minute: 0,
      hour: 9,
      months: []
    });
    expect(result).toBe('');
  });

  it('should return empty string for weekly schedule with no days selected', () => {
    const result = makeCronExpression({
      frequency: 'weekly',
      minute: 0,
      hour: 9,
      days: []
    });
    expect(result).toBe('');
  });

  it('should return empty string for monthly schedule with no days selected', () => {
    const result = makeCronExpression({
      frequency: 'monthly',
      minute: 0,
      hour: 9,
      days: []
    });
    expect(result).toBe('');
  });

  it('should generate valid yearly cron expression when months are selected', () => {
    const result = makeCronExpression({
      frequency: 'yearly',
      minute: 30,
      hour: 14,
      months: [1, 6, 12]
    });
    expect(result).toBe('30 14 1 1,6,12 *');
  });

  it('should generate valid weekly cron expression when days are selected', () => {
    const result = makeCronExpression({
      frequency: 'weekly',
      minute: 0,
      hour: 9,
      days: [1, 3, 5]
    });
    expect(result).toBe('0 9 * * 1,3,5');
  });

  it('should generate valid monthly cron expression when days are selected', () => {
    const result = makeCronExpression({
      frequency: 'monthly',
      minute: 15,
      hour: 10,
      days: [1, 15, 31]
    });
    expect(result).toBe('15 10 1,15,31 * *');
  });

  it('should generate valid daily cron expression', () => {
    const result = makeCronExpression({
      frequency: 'daily',
      minute: 30,
      hour: 8
    });
    expect(result).toBe('30 8 * * *');
  });

  it('should generate valid hourly cron expression', () => {
    const result = makeCronExpression({
      frequency: 'hourly',
      minute: 15
    });
    expect(result).toBe('15 * * * *');
  });

  it('should generate valid every minute cron expression', () => {
    const result = makeCronExpression({
      frequency: 'every minute'
    });
    expect(result).toBe('* * * * *');
  });
});